#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include <functional>
#include <thread>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <memory>

// LEMON Headers
#include <lemon/list_graph.h>
#include <lemon/smart_graph.h>
#include <lemon/preflow.h>

// Apache Arrow & Parquet Headers
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

using namespace std;
using namespace lemon;

// --------------------------- CONFIG & TYPES ---------------------------
const unsigned SAFE_NUM_THREADS = (std::thread::hardware_concurrency() == 0) ? 1u : std::thread::hardware_concurrency();
using Point = vector<double>;
using Relation = vector<Point>;
using KeyPair = pair<double, double>;
using FlatGeo = vector<vector<double>>;
using RowId = uint32_t;
using Count = long long;
using Offset = uint64_t;

static constexpr Count FLOW_INF_CAP = numeric_limits<Count>::max() / 8;
static constexpr double SEARCH_FACTOR = 1.5;
static constexpr int MAX_CELL_DIMS = 32; 

// Helper to retrieve epsilon tolerance dynamically
static double get_epsilon_tolerance() {
    const char* v = getenv("EPSILON_TOLERANCE");
    if (!v || !*v) return 1.0; // Default to 1.0 ms
    try {
        return stod(string(v));
    } catch (...) {
        return 1.0;
    }
}

// --------------------------- UTILS ---------------------------
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline uint64_t dbl_to_u64(double d) { uint64_t u; memcpy(&u, &d, sizeof(d)); return u; }
static inline double u64_to_dbl(uint64_t u) { double d; memcpy(&d, &u, sizeof(u)); return d; }

static inline Count checked_to_count(uint64_t x, const char* what) {
    if (x > (uint64_t)numeric_limits<Count>::max()) throw runtime_error(string(what) + " exceeds signed 64-bit flow capacity");
    return (Count)x;
}

static inline size_t checked_ipow_size(size_t base, int exp) {
    size_t res = 1;
    for (int i = 0; i < exp; ++i) {
        if (base != 0 && res > numeric_limits<size_t>::max() / base) throw runtime_error("k^N overflows size_t");
        res *= base;
    }
    return res;
}

class MapStringId {
    unordered_map<string, double> s2d; double nxt = 1.0;
public:
    double getId(const string& s) {
        auto it = s2d.find(s);
        if (it == s2d.end()) it = s2d.emplace(s, nxt++).first;
        return it->second;
    }
};

struct FastKeyMap {
    vector<vector<pair<uint64_t, RowId>>> table;
    size_t mask = 0;
    FastKeyMap() = default;
    explicit FastKeyMap(size_t ex) {
        size_t sz = 1;
        while (sz <= ex * 2 + 1) sz <<= 1;
        if (sz < 8) sz = 8;
        table.resize(sz);
        mask = sz - 1;
    }
    void ins(double k, RowId v) {
        const uint64_t ku = dbl_to_u64(k);
        table[splitmix64(ku) & mask].push_back({ku, v});
    }
    void get(double k, vector<RowId>& out) const {
        out.clear();
        const uint64_t ku = dbl_to_u64(k);
        const auto& bucket = table[splitmix64(ku) & mask];
        for (auto [kk, v] : bucket) if (kk == ku) out.push_back(v);
    }
};

double parseTime(const string& t) {
    if (t.size() < 19) return 0.0;
    int yy = stoi(t.substr(0, 4)), mm = stoi(t.substr(5, 2)), dd = stoi(t.substr(8, 2));
    int h = stoi(t.substr(11, 2)), m = stoi(t.substr(14, 2)), s = stoi(t.substr(17, 2));
    static const int mdays[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    auto is_leap = [](int y) { return (y % 4 == 0) && ((y % 100) != 0 || (y % 400) == 0); };
    long long days = 0;
    for (int y = 1900; y < yy; ++y) days += 365 + (is_leap(y) ? 1 : 0);
    for (int mo = 1; mo < mm; ++mo) { days += mdays[mo]; if (mo == 2 && is_leap(yy)) ++days; }
    days += (dd - 1);
    return (double)(days * 86400LL + h * 3600 + m * 60 + s) * 1000.0;
}

// Helper to convert microseconds to milliseconds
static inline double timestampToMs(int64_t micros) {
    return micros / 1000.0;
}

// Overload 1: Baseline auxiliary solver distance helper (using standard FlatGeo)
static inline double dist_sq_rows(const RowId* a, const RowId* b, int N, const FlatGeo& fg) {
    double s = 0.0;
    for (int l = 0; l < N; ++l) {
        double dx = fg[l][2 * (size_t)a[l]] - fg[l][2 * (size_t)b[l]];
        double dy = fg[l][2 * (size_t)a[l] + 1] - fg[l][2 * (size_t)b[l] + 1];
        s += dx * dx + dy * dy;
    }
    return s;
}

// Overload 2: Optimised flat pointer distance pruning exit (for hot inner-loops)
static inline double dist_sq_rows(const RowId* a, const RowId* b, int N, const vector<const double*>& fg_ptr, double th2 = numeric_limits<double>::infinity()) {
    double s = 0.0;
    for (int l = 0; l < N; ++l) {
        const double* ptr = fg_ptr[l];
        double dx = ptr[2 * (size_t)a[l]] - ptr[2 * (size_t)b[l]];
        double dy = ptr[2 * (size_t)a[l] + 1] - ptr[2 * (size_t)b[l] + 1];
        s += dx * dx + dy * dy;
        if (s > th2) return s; 
    }
    return s;
}

inline double euclid_sq(const Point &a, const Point &b) {
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) { double d = a[i] - b[i]; sum += d * d; }
    return sum;
}
inline double euclid(const Point &a, const Point &b) { return sqrt(euclid_sq(a, b)); }

static inline int color_of_layer0_row(const FlatGeo& fg, RowId id0) {
    return (fmod(fg[0][2 * (size_t)id0], 86400000.0) < 43200000.0) ? 0 : 1; // 0 red, 1 blue
}

// --------------------------- CONFIGURATION HELPERS ---------------------------
static string env_string_or_default(const char* name, const string& def) {
    const char* v = getenv(name);
    return (v && *v) ? string(v) : def;
}

static int parse_hhmm_to_minute(const string& x, int def) {
    if (x.empty()) return def;
    try {
        size_t c = x.find(':');
        int h = 0, m = 0;
        if (c == string::npos) { h = stoi(x); m = 0; }
        else { h = stoi(x.substr(0, c)); m = stoi(x.substr(c + 1)); }
        if (h < 0 || h > 24 || m < 0 || m > 59) return def;
        if (h == 24 && m != 0) return def;
        return h * 60 + m;
    } catch (...) { return def; }
}

static int minute_of_day_from_timestamp_string(const string& t) {
    if (t.size() < 16) return -1;
    try {
        int h = stoi(t.substr(11, 2));
        int m = stoi(t.substr(14, 2));
        return h * 60 + m;
    } catch (...) { return -1; }
}

static vector<pair<int,int>> parse_query_windows_or_default() {
    const string windows = env_string_or_default("QUERY_WINDOWS", "");
    vector<pair<int,int>> out;
    if (!windows.empty()) {
        size_t pos = 0;
        while (pos < windows.size()) {
            size_t comma = windows.find(',', pos);
            string part = windows.substr(pos, (comma == string::npos) ? string::npos : comma - pos);
            size_t dash = part.find('-');
            if (dash == string::npos) throw runtime_error("invalid QUERY_WINDOWS; use HH:MM-HH:MM[,HH:MM-HH:MM]");
            int a = parse_hhmm_to_minute(part.substr(0, dash), -1);
            int b = parse_hhmm_to_minute(part.substr(dash + 1), -1);
            if (a < 0 || b > 24 * 60 || a >= b) throw runtime_error("invalid QUERY_WINDOWS interval; use HH:MM-HH:MM with start < end");
            out.push_back({a,b});
            if (comma == string::npos) break;
            pos = comma + 1;
        }
        if (out.empty()) throw runtime_error("QUERY_WINDOWS parsed to zero intervals");
        return out;
    }

    const int start_min = parse_hhmm_to_minute(env_string_or_default("QUERY_START", "00:00"), 0);
    const int end_min = parse_hhmm_to_minute(env_string_or_default("QUERY_END", "24:00"), 24 * 60);
    if (start_min < 0 || end_min > 24 * 60 || start_min >= end_min) {
        throw runtime_error("invalid QUERY_START/QUERY_END; use HH:MM with QUERY_START < QUERY_END");
    }
    out.push_back({start_min, end_min});
    return out;
}

static bool minute_in_any_window(int dep_min, const vector<pair<int,int>>& windows) {
    for (auto [a,b] : windows) if (dep_min >= a && dep_min < b) return true;
    return false;
}

static string windows_to_string(const vector<pair<int,int>>& windows) {
    auto fmt = [](int x) {
        ostringstream os;
        os << setw(2) << setfill('0') << (x / 60) << ':' << setw(2) << setfill('0') << (x % 60);
        return os.str();
    };
    string out;
    for (size_t i = 0; i < windows.size(); ++i) {
        if (i) out += ",";
        out += "[" + fmt(windows[i].first) + "," + fmt(windows[i].second) + ")";
    }
    return out;
}

// --------------------------- PARQUET LOADER ---------------------------
void load_parquet(const string& fn, Relation& lg, vector<KeyPair>& lk) {
    auto file = arrow::io::ReadableFile::Open(fn).ValueOrDie();
    auto arrow_reader = parquet::arrow::OpenFile(file, arrow::default_memory_pool()).ValueOrDie();
    
    shared_ptr<arrow::Table> table;
    auto s = arrow_reader->ReadTable(&table);
    if (!s.ok()) {
        throw runtime_error("Failed to read Parquet table from: " + fn);
    }
    
    auto pickup_idx = table->schema()->GetFieldIndex("tpep_pickup_datetime");
    auto dropoff_idx = table->schema()->GetFieldIndex("tpep_dropoff_datetime");
    auto pu_idx = table->schema()->GetFieldIndex("PULocationID");
    auto do_idx = table->schema()->GetFieldIndex("DOLocationID");
    
    if (pickup_idx == -1 || dropoff_idx == -1 || pu_idx == -1 || do_idx == -1) {
        throw runtime_error("Required NYC Taxi columns not found in: " + fn);
    }
    
    const string date_filter = env_string_or_default("QUERY_DATE", "2025-01-17");
    const vector<pair<int,int>> windows = parse_query_windows_or_default();
    const string origin_filter = env_string_or_default("QUERY_ORIGIN", "");
    const string dest_filter = env_string_or_default("QUERY_DEST", "");
    
    double origin_val = -1.0;
    if (!origin_filter.empty()) {
        try { origin_val = stod(origin_filter); } catch (...) {}
    }
    double dest_val = -1.0;
    if (!dest_filter.empty()) {
        try { dest_val = stod(dest_filter); } catch (...) {}
    }
    
    int filter_day = -1;
    if (!date_filter.empty()) {
        if (date_filter.size() >= 10 && date_filter[4] == '-' && date_filter[7] == '-') {
            try {
                filter_day = stoi(date_filter.substr(8, 2));
            } catch (...) {}
        } else {
            try {
                filter_day = stoi(date_filter);
            } catch (...) {}
        }
    }
    
    cout << "Loading Parquet file " << fn << " with active filters: date=" << date_filter
         << " (filtering day=" << filter_day << ") windows=" << windows_to_string(windows);
    if (!origin_filter.empty()) cout << " origin=" << origin_filter;
    if (!dest_filter.empty()) cout << " dest=" << dest_filter;
    cout << endl;
    
    auto pickup_col = table->column(pickup_idx);
    auto dropoff_col = table->column(dropoff_idx);
    auto pu_col = table->column(pu_idx);
    auto do_col = table->column(do_idx);
    
    for (int c = 0; c < pickup_col->num_chunks(); ++c) {
        auto pickup_arr = std::static_pointer_cast<arrow::TimestampArray>(pickup_col->chunk(c));
        auto dropoff_arr = std::static_pointer_cast<arrow::TimestampArray>(dropoff_col->chunk(c));
        auto pu_arr = std::static_pointer_cast<arrow::Int64Array>(pu_col->chunk(c));
        auto do_arr = std::static_pointer_cast<arrow::Int64Array>(do_col->chunk(c));
        
        int64_t chunk_length = pickup_arr->length();
        for (int64_t i = 0; i < chunk_length; ++i) {
            if (pickup_arr->IsNull(i) || dropoff_arr->IsNull(i) || pu_arr->IsNull(i) || do_arr->IsNull(i)) {
                continue;
            }
            
            int64_t pickup_micros = pickup_arr->Value(i);
            time_t rt = pickup_micros / 1000000;
            struct tm *ptm = gmtime(&rt);
            if (ptm == nullptr) continue;
            
            int dy = ptm->tm_mday;
            
            if (filter_day != -1 && dy != filter_day) {
                continue;
            }
            
            int dep_min = ptm->tm_hour * 60 + ptm->tm_min;
            if (!minute_in_any_window(dep_min, windows)) continue;
            
            double p_val = static_cast<double>(pu_arr->Value(i));
            double d_val = static_cast<double>(do_arr->Value(i));
            
            if (origin_val != -1.0 && p_val != origin_val) continue;
            if (dest_val != -1.0 && d_val != dest_val) continue;
            
            double pickup_ms = timestampToMs(pickup_micros);
            double dropoff_ms = timestampToMs(dropoff_arr->Value(i));
            
            lg.push_back({pickup_ms, dropoff_ms});
            lk.emplace_back(p_val, d_val);
        }
    }
    
    if (lg.empty()) {
        throw runtime_error("Query filters selected 0 rows from Parquet dataset: " + fn);
    }
    
    size_t input_red = 0, input_blue = 0;
    for (const Point& p : lg) {
        if (fmod(p[0], 86400000.0) < 43200000.0) ++input_red;
        else ++input_blue;
    }
    cout << "Completed loading " << lg.size() << " trips from " << fn 
         << " | input_red_rows=" << input_red << " input_blue_rows=" << input_blue << endl;
}

static int env_int_or_default(const char* name, int def) {
    const char* v = getenv(name);
    if (!v || !*v) return def;
    try {
        return stoi(string(v));
    } catch (...) {
        cerr << "Ignoring invalid integer environment variable " << name << "=" << v << endl;
        return def;
    }
}

static bool env_flag_is_one(const char* name) {
    const char* v = getenv(name);
    return v && string(v) == "1";
}

static Count env_count_or_default(const char* name, Count def) {
    const char* v = getenv(name);
    if (!v || !*v) return def;
    try {
        long long x = stoll(string(v));
        if (x < 0) return def;
        return (Count)x;
    } catch (...) {
        cerr << "Ignoring invalid integer environment variable " << name << "=" << v << endl;
        return def;
    }
}

// --------------------------- GRAPH LAYER ---------------------------
struct GLayer {
    vector<Offset> off;
    vector<RowId> tgt;
};

// --------------------------- CORESET RESULT ---------------------------
struct CoresetResult {
    vector<Point> red_centers;
    vector<Point> blue_centers;
    vector<Count> red_weights;
    vector<Count> blue_weights;
    Count count_red = 0;
    Count count_blue = 0;

    int N = 0;
    int k = 0;
    size_t total_cells = 0;
    vector<vector<Point>> layer_centers;
    vector<vector<int>> n2c;
    vector<int> red_cell_to_idx;
    vector<int> blue_cell_to_idx;
};

struct Fairlet {
    int blue_idx = -1;
    vector<pair<int, Count>> red_flow; 
    Point rep;                         
    unordered_map<int, Count> f_r; // assigned flow weights per red representative
};

struct FairletSolution {
    bool feasible = false;
    double threshold = 0.0;
    Count flow_value = 0;
    Count total_red = 0;
    Count total_blue = 0;
    vector<Fairlet> fairlets;
    vector<int> blue_to_fairlet;
    vector<vector<pair<int, Count>>> red_to_fairlets;
    vector<Point> centers;             
};

// --------------------------- CORE COMPONENTS ---------------------------
static vector<int> gonz_fl(const double* pts, size_t n, int k) {
    if (n == 0 || k <= 0) return {};
    vector<int> c = {0};
    vector<double> md(n, numeric_limits<double>::infinity());
    for (int it = 1; it < min(k, (int)n); ++it) {
        double mx = -1; int b = 0; const double* cp = &pts[(size_t)c.back() * 2];
        for (size_t i = 0; i < n; ++i) {
            double dx = pts[i * 2] - cp[0], dy = pts[i * 2 + 1] - cp[1], ds = dx * dx + dy * dy;
            if (ds < md[i]) md[i] = ds;
            if (md[i] > mx) { mx = md[i]; b = (int)i; }
        }
        c.push_back(b);
    }
    return c;
}

vector<Point> run_unfair_gonzales(const vector<Point>& pts, int k_target) {
    if (pts.empty() || k_target <= 0) return {};
    int n = (int)pts.size(); int k = min(k_target, n);
    vector<Point> centers; centers.reserve(k); centers.push_back(pts[0]);
    vector<double> min_sq_dist(n, numeric_limits<double>::infinity());
    for (int i = 1; i < k; ++i) {
        const Point& lc = centers.back(); int best_p = 0; double mx = -1.0;
        for (int p = 0; p < n; ++p) {
            double sq = euclid_sq(pts[p], lc);
            if (sq < min_sq_dist[p]) min_sq_dist[p] = sq;
            if (min_sq_dist[p] > mx) { mx = min_sq_dist[p]; best_p = p; }
        }
        centers.push_back(pts[best_p]);
    }
    return centers;
}

static inline int closest_center_idx(const Point& p, const vector<Point>& centers) {
    int best = -1; double bd = numeric_limits<double>::infinity();
    for (int i = 0; i < (int)centers.size(); ++i) {
        double d = euclid_sq(p, centers[i]);
        if (d < bd) { bd = d; best = i; }
    }
    return best;
}

// --------------------------- APPROX WORKFLOW ---------------------------
CoresetResult run_approx_workflow_weighted(const FlatGeo& fg, const vector<GLayer>& gr, int k) {
    int N = (int)fg.size();
    if (N <= 0 || k <= 0) throw runtime_error("invalid N/k in coreset workflow");

    CoresetResult res;
    res.N = N; res.k = k;
    res.layer_centers.assign(N, {});

    for (int i = 0; i < N; i++) {
        vector<int> c_idx = gonz_fl(fg[i].data(), fg[i].size() / 2, k);
        res.layer_centers[i].reserve(c_idx.size());
        for (int idx : c_idx) res.layer_centers[i].push_back({fg[i][2 * (size_t)idx], fg[i][2 * (size_t)idx + 1]});
        if ((int)res.layer_centers[i].size() != k) throw runtime_error("a layer has fewer than k points");
    }

    res.n2c.assign(N, {});
    for (int i = 0; i < N; i++) {
        size_t npts = fg[i].size() / 2;
        res.n2c[i].resize(npts);
        for (size_t p = 0; p < npts; p++) {
            double px = fg[i][2 * p], py = fg[i][2 * p + 1], min_d = numeric_limits<double>::infinity(); int b = 0;
            for (int c = 0; c < (int)res.layer_centers[i].size(); c++) {
                double dx = px - res.layer_centers[i][c][0], dy = py - res.layer_centers[i][c][1];
                double d = dx * dx + dy * dy;
                if (d < min_d) { min_d = d; b = c; }
            }
            res.n2c[i][p] = b;
        }
    }

    res.total_cells = checked_ipow_size((size_t)k, N);
    if (res.total_cells > numeric_limits<uint32_t>::max()) throw runtime_error("too many product cells for uint32_t coreset-count DP codes");
    vector<Count> r_w(res.total_cells, 0), b_w(res.total_cells, 0);

    vector<size_t> digit_mul((size_t)N, 1);
    for (int l = N - 2; l >= 0; --l) digit_mul[(size_t)l] = digit_mul[(size_t)l + 1] * (size_t)k;

    struct CountRowDP {
        vector<uint32_t> code; 
        vector<Count> cnt;     
    };

    const int last = N - 1;
    if (N == 1) {
        for (size_t u = 0; u < fg[0].size() / 2; ++u) {
            size_t cell = (size_t)res.n2c[0][u];
            if (color_of_layer0_row(fg, (RowId)u) == 0) r_w[cell]++;
            else b_w[cell]++;
        }
    } else {
        vector<CountRowDP> next(fg[(size_t)last].size() / 2);
        for (size_t u = 0; u < next.size(); ++u) {
            next[u].code.push_back((uint32_t)res.n2c[(size_t)last][u]);
            next[u].cnt.push_back(1);
        }

        for (int l = N - 2; l >= 1; --l) {
            const size_t rows = fg[(size_t)l].size() / 2;
            vector<CountRowDP> cur(rows);
            size_t total_states = 0;
            for (size_t u = 0; u < rows; ++u) {
                CountRowDP& out = cur[u];
                unordered_map<uint32_t, size_t> pos;
                pos.reserve(256);
                const uint32_t prefix = (uint32_t)((size_t)res.n2c[(size_t)l][u] * digit_mul[(size_t)l]);
                const Offset s = gr[(size_t)l].off[u], e = gr[(size_t)l].off[u + 1];
                for (Offset a = s; a < e; ++a) {
                    const RowId v = gr[(size_t)l].tgt[a];
                    const CountRowDP& nd = next[(size_t)v];
                    for (size_t si = 0; si < nd.code.size(); ++si) {
                        const uint32_t code = prefix + nd.code[si];
                        auto it = pos.find(code);
                        size_t p;
                        if (it == pos.end()) {
                            p = out.code.size();
                            pos.emplace(code, p);
                            out.code.push_back(code);
                            out.cnt.push_back(0);
                        } else {
                            p = it->second;
                        }
                        if (out.cnt[p] > numeric_limits<Count>::max() - nd.cnt[si]) throw runtime_error("coreset cell count overflows int64");
                        out.cnt[p] += nd.cnt[si];
                    }
                }
                total_states += out.code.size();
            }
            cout << "Compressed coreset-count DP layer " << l << " states=" << total_states << endl;
            next.swap(cur);
        }

        size_t final_transitions = 0;
        for (size_t u = 0; u < fg[0].size() / 2; ++u) {
            const uint32_t prefix = (uint32_t)((size_t)res.n2c[0][u] * digit_mul[0]);
            vector<Count>& dst = (color_of_layer0_row(fg, (RowId)u) == 0) ? r_w : b_w;
            const Offset s = gr[0].off[u], e = gr[0].off[u + 1];
            for (Offset a = s; a < e; ++a) {
                const RowId v = gr[0].tgt[a];
                const CountRowDP& nd = next[(size_t)v];
                for (size_t si = 0; si < nd.code.size(); ++si) {
                    const size_t cell = (size_t)(prefix + nd.code[si]);
                    if (dst[cell] > numeric_limits<Count>::max() - nd.cnt[si]) throw runtime_error("coreset full cell count overflows int64");
                    dst[cell] += nd.cnt[si];
                    ++final_transitions;
                }
            }
        }
        cout << "Compressed coreset-count DP final transitions=" << final_transitions << endl;
    }

    res.red_cell_to_idx.assign(res.total_cells, -1);
    res.blue_cell_to_idx.assign(res.total_cells, -1);

    for (size_t cell = 0; cell < res.total_cells; ++cell) {
        if (r_w[cell] == 0 && b_w[cell] == 0) continue;
        Point p(2 * (size_t)N);
        size_t tk = cell;
        for (int l = N - 1; l >= 0; --l) {
            int ci = (int)(tk % (size_t)k);
            p[2 * (size_t)l] = res.layer_centers[l][ci][0];
            p[2 * (size_t)l + 1] = res.layer_centers[l][ci][1];
            tk /= (size_t)k;
        }
        if (r_w[cell] > 0) {
            res.red_cell_to_idx[cell] = (int)res.red_centers.size();
            res.red_centers.push_back(p);
            res.red_weights.push_back(r_w[cell]);
            res.count_red += r_w[cell];
        }
        if (b_w[cell] > 0) {
            res.blue_cell_to_idx[cell] = (int)res.blue_centers.size();
            res.blue_centers.push_back(p);
            res.blue_weights.push_back(b_w[cell]);
            res.count_blue += b_w[cell];
        }
    }
    return res;
}

// --------------------------- POINT GRID FOR WEIGHTED FLOW EDGE GENERATION ---------------------------
struct PointCellKey {
    int d = 0;
    array<int64_t, MAX_CELL_DIMS> x{};
    bool operator==(const PointCellKey& other) const {
        if (d != other.d) return false;
        for (int i = 0; i < d; ++i) if (x[i] != other.x[i]) return false;
        return true;
    }
};

// Optimised 64-bit FNV-1a Hash Function
struct PointCellKeyHash {
    size_t operator()(const PointCellKey& k) const {
        uint64_t h = 14695981039346656037ULL;
        for (int i = 0; i < k.d; ++i) {
            h ^= static_cast<uint64_t>(k.x[i]);
            h *= 1099511628211ULL;
        }
        return static_cast<size_t>(h);
    }
};

static PointCellKey cell_of_point(const Point& p, double side) {
    if ((int)p.size() > MAX_CELL_DIMS) throw runtime_error("too many dimensions for PointCellKey; increase MAX_CELL_DIMS");
    PointCellKey key; key.d = (int)p.size();
    for (int i = 0; i < key.d; ++i) key.x[i] = (int64_t)floor(p[i] / side);
    return key;
}

static void gen_point_neighbor_offsets_rec(int pos, int d, array<int8_t, MAX_CELL_DIMS>& cur,
                                           vector<array<int8_t, MAX_CELL_DIMS>>& out) {
    if (pos == d) { out.push_back(cur); return; }
    for (int z = -1; z <= 1; ++z) { cur[pos] = (int8_t)z; gen_point_neighbor_offsets_rec(pos + 1, d, cur, out); }
}

static vector<array<int8_t, MAX_CELL_DIMS>> gen_point_neighbor_offsets(int d) {
    if (d > MAX_CELL_DIMS) throw runtime_error("too many dimensions for point neighbor offsets");
    vector<array<int8_t, MAX_CELL_DIMS>> out;
    array<int8_t, MAX_CELL_DIMS> cur{};
    gen_point_neighbor_offsets_rec(0, d, cur, out);
    return out;
}

static inline PointCellKey shifted_point_key(const PointCellKey& base, const array<int8_t, MAX_CELL_DIMS>& off) {
    PointCellKey y = base;
    for (int i = 0; i < y.d; ++i) y.x[i] += off[i];
    return y;
}

FairletSolution solve_weighted_fairlets_naive(const vector<Point>& red_pts,
                                              const vector<Count>& red_w,
                                              const vector<Point>& blue_pts,
                                              const vector<Count>& blue_w,
                                              double threshold,
                                              int k_final) {
    FairletSolution sol;
    sol.threshold = threshold;
    sol.total_red = accumulate(red_w.begin(), red_w.end(), (Count)0);
    sol.total_blue = accumulate(blue_w.begin(), blue_w.end(), (Count)0);

    const int nr = (int)red_pts.size(), nb = (int)blue_pts.size();
    sol.blue_to_fairlet.assign(nb, -1);
    sol.red_to_fairlets.assign(nr, {});

    if (nr == 0 || nb == 0 || sol.total_red == 0) {
        sol.feasible = false;
        return sol;
    }

    ListDigraph g;
    ListDigraph::ArcMap<Count> cap(g);
    vector<ListDigraph::Node> r_nodes(nr), b_nodes(nb);
    ListDigraph::Node src = g.addNode(), snk = g.addNode();
    for (int j = 0; j < nr; ++j) { r_nodes[j] = g.addNode(); cap[g.addArc(src, r_nodes[j])] = red_w[j]; }
    const Count infcap = min(sol.total_red, FLOW_INF_CAP);
    for (int i = 0; i < nb; ++i) { b_nodes[i] = g.addNode(); cap[g.addArc(b_nodes[i], snk)] = infcap; }

    vector<pair<int, int>> edge_idx; 
    vector<ListDigraph::Arc> rb_arcs;
    const double th2 = threshold * threshold;

    unordered_map<PointCellKey, vector<int>, PointCellKeyHash> blue_grid;
    blue_grid.reserve((size_t)nb * 2 + 1);
    for (int i = 0; i < nb; ++i) blue_grid[cell_of_point(blue_pts[i], threshold)].push_back(i);

    const int dim = red_pts.empty() ? 0 : (int)red_pts[0].size();
    const auto offsets = gen_point_neighbor_offsets(dim);
    for (int j = 0; j < nr; ++j) {
        PointCellKey rc = cell_of_point(red_pts[j], threshold);
        for (const auto& off : offsets) {
            PointCellKey nc = shifted_point_key(rc, off);
            auto mt = blue_grid.find(nc);
            if (mt == blue_grid.end()) continue;
            for (int i : mt->second) {
                if (euclid_sq(red_pts[j], blue_pts[i]) <= th2) {
                    auto a = g.addArc(r_nodes[j], b_nodes[i]);
                    cap[a] = infcap;
                    rb_arcs.push_back(a);
                    edge_idx.push_back({j, i});
                }
            }
        }
    }

    Preflow<ListDigraph, ListDigraph::ArcMap<Count>> pf(g, cap, src, snk);
    pf.run();
    sol.flow_value = pf.flowValue();
    sol.feasible = (sol.flow_value == sol.total_red);
    if (!sol.feasible) return sol;

    vector<vector<pair<int, Count>>> incoming(nb);
    for (size_t e = 0; e < rb_arcs.size(); ++e) {
        Count f = pf.flow(rb_arcs[e]);
        if (f <= 0) continue;
        int r = edge_idx[e].first, b = edge_idx[e].second;
        incoming[b].push_back({r, f});
    }

    vector<Point> fairlet_reps;
    fairlet_reps.reserve(nb);
    sol.fairlets.reserve(nb);
    for (int b = 0; b < nb; ++b) {
        int fid = (int)sol.fairlets.size();
        sol.blue_to_fairlet[b] = fid;
        Fairlet fl;
        fl.blue_idx = b;
        fl.red_flow = std::move(incoming[b]);
        fl.rep = blue_pts[b];
        for (auto [r, f] : fl.red_flow) {
            sol.red_to_fairlets[r].push_back({fid, f});
            fl.f_r[r] = f; 
        }
        fairlet_reps.push_back(fl.rep);
        sol.fairlets.push_back(std::move(fl));
    }

    sol.centers = run_unfair_gonzales(fairlet_reps, k_final);
    return sol;
}

FairletSolution run_weighted_threshold_search(const vector<Point>& red_pts,
                                              const vector<Count>& red_w,
                                              const vector<Point>& blue_pts,
                                              const vector<Count>& blue_w,
                                              double initial_r,
                                              int k_final) {
    double r = initial_r;
    FairletSolution best_sol;
    bool found_any = false;
    
    double lo_bad = 0.0;
    double hi_good = numeric_limits<double>::infinity();
    
    const double epsilon_tol = get_epsilon_tolerance();
    
    while (true) {
        if (hi_good - lo_bad <= epsilon_tol && hi_good != numeric_limits<double>::infinity()) {
            break;
        }
        
        FairletSolution cur = solve_weighted_fairlets_naive(red_pts, red_w, blue_pts, blue_w, r, k_final);
        
        if (cur.feasible) {
            best_sol = std::move(cur);
            found_any = true;
            hi_good = r;
            r = r / 1.5;
        } else {
            lo_bad = r;
            r = r * 1.5;
        }
        
        if (hi_good != numeric_limits<double>::infinity() && lo_bad > 0.0) {
            r = (lo_bad + hi_good) * 0.5;
        }
    }
    
    if (!found_any) {
        throw runtime_error("No feasible threshold found during approximate coreset search");
    }
    return best_sol;
}

static size_t packed_cell_for_path(const vector<RowId>& path, const CoresetResult& core) {
    size_t pk = 0;
    for (int l = 0; l < core.N; ++l) pk = pk * (size_t)core.k + (size_t)core.n2c[l][path[l]];
    return pk;
}

// conference specified exact cost-measurement procedure
double compute_exact_clustering_cost(const FlatGeo& fg,
                                     const vector<GLayer>& gr,
                                     const CoresetResult& core,
                                     const FairletSolution& sol) {
    if (!sol.feasible || sol.centers.empty()) throw runtime_error("cannot compute cost from an infeasible/empty approximate solution");

    vector<Fairlet> fairlets_copy = sol.fairlets;
    int num_red_reps = core.red_centers.size();
    vector<vector<int>> red_to_fids(num_red_reps);
    for (int fid = 0; fid < (int)fairlets_copy.size(); ++fid) {
        for (auto const& [r_idx, f_val] : fairlets_copy[fid].red_flow) {
            if (f_val > 0) {
                red_to_fids[r_idx].push_back(fid);
            }
        }
    }

    vector<RowId> path(core.N, 0);
    Point tuple_point(2 * (size_t)core.N);
    double max_dist = 0.0;
    mutex mtx;

    function<void(int, RowId)> dfs = [&](int l, RowId id) {
        path[l] = id;
        tuple_point[2 * (size_t)l] = fg[l][2 * (size_t)id];
        tuple_point[2 * (size_t)l + 1] = fg[l][2 * (size_t)id + 1];
        if (l == core.N - 1) {
            size_t cell = packed_cell_for_path(path, core);
            int col = color_of_layer0_row(fg, path[0]);
            
            int found_fid = -1;
            if (col == 0) { 
                int p = core.red_cell_to_idx[cell];
                if (p < 0) throw runtime_error("red join tuple maps to an empty red coreset cell");
                
                for (int fid : red_to_fids[p]) {
                    if (fairlets_copy[fid].f_r[p] > 0) {
                        found_fid = fid;
                        fairlets_copy[fid].f_r[p]--;
                        break;
                    }
                }
                if (found_fid < 0) throw runtime_error("Could not find a valid fairlet for red representative");
            } 
            else { 
                int p = core.blue_cell_to_idx[cell];
                if (p < 0) throw runtime_error("blue join tuple maps to an empty blue coreset cell");
                
                found_fid = sol.blue_to_fairlet[p];
                if (found_fid < 0) throw runtime_error("Could not find a valid fairlet for blue representative");
            }

            const Point& f_rep = fairlets_copy[found_fid].rep;
            int c_idx = closest_center_idx(f_rep, sol.centers);
            const Point& c_t = sol.centers[c_idx];

            double dist = euclid(tuple_point, c_t);

            lock_guard<mutex> lk(mtx);
            if (dist > max_dist) {
                max_dist = dist;
            }
            return;
        }
        Offset s = gr[l].off[id], e = gr[l].off[(size_t)id + 1];
        for (Offset j = s; j < e; ++j) dfs(l + 1, gr[l].tgt[j]);
    };

    for (size_t i = 0; i < fg[0].size() / 2; ++i) dfs(0, (RowId)i);
    return max_dist;
}

// --------------------------- EXACT JOIN COUNTING WITHOUT MATERIALIZATION ---------------------------
struct ExactJoinCounts {
    Count red = 0;
    Count blue = 0;
    Count total = 0;
};

ExactJoinCounts count_full_join_exact(const FlatGeo& fg, const vector<GLayer>& gr) {
    const int N = (int)fg.size();
    if (N == 0) return {};

    vector<vector<Count>> dp(N);
    dp[N - 1].assign(fg[N - 1].size() / 2, 1);
    for (int l = N - 2; l >= 0; --l) {
        const size_t n = fg[l].size() / 2;
        dp[l].assign(n, 0);
        for (size_t i = 0; i < n; ++i) {
            Count s = 0;
            for (Offset e = gr[l].off[i]; e < gr[l].off[i + 1]; ++e) {
                RowId v = gr[l].tgt[e];
                if (dp[l + 1][v] > numeric_limits<Count>::max() - s) throw runtime_error("exact join count overflows int64");
                s += dp[l + 1][v];
            }
            dp[l][i] = s;
        }
    }

    ExactJoinCounts out;
    for (size_t i = 0; i < dp[0].size(); ++i) {
        if (color_of_layer0_row(fg, (RowId)i) == 0) out.red += dp[0][i];
        else out.blue += dp[0][i];
    }
    out.total = out.red + out.blue;
    return out;
}

static constexpr Count DEFAULT_EXACT_BASELINE_TUPLE_LIMIT = 50000000LL;

// --------------------------- COMPACT FULL-JOIN BASELINE ---------------------------
struct CompactJoinStore {
    int N = 0;
    vector<RowId> red_rows;   
    vector<RowId> blue_rows;
    size_t red_count() const { return N == 0 ? 0 : red_rows.size() / (size_t)N; }
    size_t blue_count() const { return N == 0 ? 0 : blue_rows.size() / (size_t)N; }
    const RowId* red_tuple(size_t i) const { return &red_rows[i * (size_t)N]; }
    const RowId* blue_tuple(size_t i) const { return &blue_rows[i * (size_t)N]; }
};

static inline void append_tuple(vector<RowId>& dst, const vector<RowId>& path) {
    dst.insert(dst.end(), path.begin(), path.end());
}

CompactJoinStore enumerate_full_join_compact(const FlatGeo& fg, const vector<GLayer>& gr) {
    const int N = (int)fg.size();
    CompactJoinStore out; out.N = N;
    const size_t roots = fg[0].size() / 2;
    const unsigned T = min<unsigned>(SAFE_NUM_THREADS, (unsigned)max<size_t>(1, roots));

    vector<CompactJoinStore> locals(T);
    vector<thread> workers;
    atomic<size_t> next_root(0);

    auto worker = [&](unsigned tid) {
        locals[tid].N = N;
        vector<RowId> path(N, 0);
        function<void(int, RowId)> dfs = [&](int l, RowId id) {
            path[l] = id;
            if (l == N - 1) {
                int col = color_of_layer0_row(fg, path[0]);
                if (col == 0) append_tuple(locals[tid].red_rows, path);
                else append_tuple(locals[tid].blue_rows, path);
                return;
            }
            Offset s = gr[l].off[id], e = gr[l].off[(size_t)id + 1];
            for (Offset j = s; j < e; ++j) dfs(l + 1, gr[l].tgt[j]);
        };
        while (true) {
            size_t r = next_root.fetch_add(1);
            if (r >= roots) break;
            dfs(0, (RowId)r);
        }
    };

    for (unsigned t = 0; t < T; ++t) workers.emplace_back(worker, t);
    for (auto& th : workers) th.join();

    size_t red_total = 0, blue_total = 0;
    for (auto& s : locals) { red_total += s.red_rows.size(); blue_total += s.blue_rows.size(); }
    out.red_rows.reserve(red_total);
    out.blue_rows.reserve(blue_total);
    for (auto& s : locals) {
        out.red_rows.insert(out.red_rows.end(), s.red_rows.begin(), s.red_rows.end());
        out.blue_rows.insert(out.blue_rows.end(), s.blue_rows.begin(), s.blue_rows.end());
    }
    return out;
}

static inline double dist_sq_rows_to_point(const RowId* a, int N, const FlatGeo& fg, const Point& p) {
    double s = 0.0;
    for (int l = 0; l < N; ++l) {
        double dx = fg[l][2 * (size_t)a[l]] - p[2 * (size_t)l];
        double dy = fg[l][2 * (size_t)a[l] + 1] - p[2 * (size_t)l + 1];
        s += dx * dx + dy * dy;
    }
    return s;
}

static inline Point materialize_tuple_point(const RowId* rows, int N, const FlatGeo& fg) {
    Point p(2 * (size_t)N);
    for (int l = 0; l < N; ++l) {
        p[2 * (size_t)l] = fg[l][2 * (size_t)rows[l]];
        p[2 * (size_t)l + 1] = fg[l][2 * (size_t)rows[l] + 1];
    }
    return p;
}

struct CellKey {
    int d = 0;
    array<int64_t, MAX_CELL_DIMS> x{};
    bool operator==(const CellKey& other) const {
        if (d != other.d) return false;
        for (int i = 0; i < d; ++i) if (x[i] != other.x[i]) return false;
        return true;
    }
};

// Optimised 64-bit FNV-1a Hash Function
struct CellKeyHash {
    size_t operator()(const CellKey& k) const {
        uint64_t h = 14695981039346656037ULL;
        for (int i = 0; i < k.d; ++i) {
            h ^= static_cast<uint64_t>(k.x[i]);
            h *= 1099511628211ULL;
        }
        return static_cast<size_t>(h);
    }
};

static CellKey cell_of_tuple(const RowId* rows, int N, const FlatGeo& fg, double side) {
    const int d = 2 * N;
    if (d > MAX_CELL_DIMS) throw runtime_error("too many dimensions for CellKey; increase MAX_CELL_DIMS");
    CellKey key; key.d = d;
    for (int l = 0; l < N; ++l) {
        key.x[2 * l] = (int64_t)floor(fg[l][2 * (size_t)rows[l]] / side);
        key.x[2 * l + 1] = (int64_t)floor(fg[l][2 * (size_t)rows[l] + 1] / side);
    }
    return key;
}

static void gen_neighbor_offsets_rec(int pos, int d, array<int8_t, MAX_CELL_DIMS>& cur,
                                     vector<array<int8_t, MAX_CELL_DIMS>>& out) {
    if (pos == d) { out.push_back(cur); return; }
    for (int z = -1; z <= 1; ++z) { cur[pos] = (int8_t)z; gen_point_neighbor_offsets_rec(pos + 1, d, cur, out); }
}

static vector<array<int8_t, MAX_CELL_DIMS>> gen_neighbor_offsets(int d) {
    if (d > MAX_CELL_DIMS) throw runtime_error("too many dimensions for neighbor offsets");
    vector<array<int8_t, MAX_CELL_DIMS>> out;
    array<int8_t, MAX_CELL_DIMS> cur{};
    gen_neighbor_offsets_rec(0, d, cur, out);
    return out;
}

static inline CellKey shifted_key(const CellKey& base, const array<int8_t, MAX_CELL_DIMS>& off) {
    CellKey y = base;
    for (int i = 0; i < y.d; ++i) y.x[i] += off[i];
    return y;
}

struct CompactFairlet {
    RowId blue_idx = -1;
    vector<pair<RowId, Count>> red_flow; // red tuple ID, flow
    Point rep;
    unordered_map<RowId, Count> f_r;     // Map to store assigned flows for red tuples
};

struct CompactFairletSolution {
    bool feasible = false;
    double threshold = 0.0;
    Count flow_value = 0;
    Count total_red = 0;
    Count total_blue = 0;
    vector<int> blue_to_fairlet;                       
    vector<vector<pair<int, Count>>> red_to_fairlets;  
    vector<RowId> fairlet_blue_idx;                    
    vector<CompactFairlet> fairlets; 
    vector<Point> centers;                             
};

static vector<Point> run_gonzales_on_compact_fairlet_reps(const CompactJoinStore& store,
                                                          const FlatGeo& fg,
                                                          const vector<RowId>& fairlet_blue_idx,
                                                          int k_final) {
    const size_t m = fairlet_blue_idx.size();
    if (m == 0 || k_final <= 0) return {};
    const int k = min<int>(k_final, (int)m);
    vector<int> chosen; chosen.reserve(k); chosen.push_back(0);
    vector<double> md(m, numeric_limits<double>::infinity());

    // Corrected off-by-one pre-increment loop index bug inside baseline Gonzales solver
    for (int it = 1; it < k; ++it) {
        const RowId center_b = fairlet_blue_idx[(size_t)chosen.back()];
        const RowId* cp = store.blue_tuple(center_b);
        double mx = -1.0; int best = 0;
        for (size_t i = 0; i < m; ++i) {
            const RowId* pp = store.blue_tuple(fairlet_blue_idx[i]);
            double ds = dist_sq_rows(pp, cp, store.N, fg);
            if (ds < md[i]) md[i] = ds;
            if (md[i] > mx) { mx = md[i]; best = (int)i; }
        }
        chosen.push_back(best);
    }

    vector<Point> centers; centers.reserve(chosen.size());
    for (int idx : chosen) centers.push_back(materialize_tuple_point(store.blue_tuple(fairlet_blue_idx[(size_t)idx]), store.N, fg));
    return centers;
}

struct CandidateEdge { RowId r; RowId b; };

// Exact baseline solver, updated with parallel lock-free greedy pre-checks, SmartDigraph,
// and high-performance pre-cached flat pointer geometry lookups.
CompactFairletSolution solve_full_join_baseline_at_r(const CompactJoinStore& store,
                                                     const vector<const double*>& fg_ptr,
                                                     const FlatGeo& fg,
                                                     double threshold,
                                                     int k_final) {
    CompactFairletSolution sol;
    sol.threshold = threshold;
    const size_t R = store.red_count(), B = store.blue_count();
    sol.total_red = checked_to_count(R, "red join count");
    sol.total_blue = checked_to_count(B, "blue join count");
    sol.blue_to_fairlet.assign(B, -1);
    sol.red_to_fairlets.assign(R, {});

    if (R == 0 || B == 0 || sol.total_red == 0 || threshold <= 0.0) return sol;
    if (R > numeric_limits<RowId>::max() || B > numeric_limits<RowId>::max()) throw runtime_error("too many exact join tuples for RowId indexing");

    const Count infcap = min(sol.total_red, FLOW_INF_CAP);
    const double th2 = threshold * threshold;

    // TIMER 1: Parallel Greedy matching pre-check
    auto t_greedy_start = chrono::high_resolution_clock::now();

    auto blue_matched = std::make_unique<std::atomic<bool>[]>(B);
    for (size_t b = 0; b < B; ++b) blue_matched[b].store(false, memory_order_relaxed);

    vector<int> red_to_blue_match(R, -1);
    atomic<size_t> greedy_matches(0);
    atomic<size_t> next_r(0);
    const size_t G_CHUNK = 128;

    const unsigned T_greedy = min<unsigned>(SAFE_NUM_THREADS, (unsigned)max<size_t>(1, R / G_CHUNK + 1));
    vector<thread> greedy_threads;

    auto greedy_worker = [&]() {
        while (true) {
            size_t r0 = next_r.fetch_add(G_CHUNK, memory_order_relaxed);
            if (r0 >= R) break;
            size_t r1 = min(R, r0 + G_CHUNK);
            for (size_t r = r0; r < r1; ++r) {
                const RowId* rt = store.red_tuple(r);
                for (size_t b = 0; b < B; ++b) {
                    bool expected = false;
                    if (blue_matched[b].compare_exchange_strong(expected, true, memory_order_relaxed)) {
                        // Utilising fast cached pointer array
                        if (dist_sq_rows(rt, store.blue_tuple(b), store.N, fg_ptr, th2) <= th2) {
                            red_to_blue_match[r] = (int)b;
                            greedy_matches.fetch_add(1, memory_order_relaxed);
                            break;
                        } else {
                            blue_matched[b].store(false, memory_order_relaxed);
                        }
                    }
                }
            }
        }
    };

    for (unsigned t = 0; t < T_greedy; ++t) greedy_threads.emplace_back(greedy_worker);
    for (auto& th : greedy_threads) th.join();

    auto t_greedy_end = chrono::high_resolution_clock::now();

    // If the greedy match found is a perfect matching, bypass LEMON completely
    if (greedy_matches == R) {
        long long d_greedy = chrono::duration_cast<chrono::milliseconds>(t_greedy_end - t_greedy_start).count();
        
        vector<vector<pair<RowId, Count>>> incoming(B);
        for (size_t r = 0; r < R; ++r) {
            incoming[(size_t)red_to_blue_match[r]].push_back({(RowId)r, 1});
        }
        
        sol.fairlets.reserve(B);
        for (size_t b = 0; b < B; ++b) {
            int fid = (int)sol.fairlet_blue_idx.size();
            sol.blue_to_fairlet[b] = fid;
            sol.fairlet_blue_idx.push_back((RowId)b);
            
            CompactFairlet fl;
            fl.blue_idx = (RowId)b;
            fl.red_flow = std::move(incoming[b]);
            fl.rep = materialize_tuple_point(store.blue_tuple(b), store.N, fg);
            for (auto [r, f] : fl.red_flow) {
                sol.red_to_fairlets[r].push_back({fid, f});
                fl.f_r[r] = f;
            }
            sol.fairlets.push_back(std::move(fl));
        }

        sol.centers = run_gonzales_on_compact_fairlet_reps(store, fg, sol.fairlet_blue_idx, k_final);
        sol.flow_value = R;
        sol.feasible = true;

        cout << "  [solve_at_r=" << threshold << "] FEASIBLE (Greedy Pruned) | "
             << "Greedy Time: " << d_greedy << "ms" << endl;
        return sol;
    }

    // --- FALLBACK TO FULL GRAPH BUILD & MAX-FLOW (Only if greedy fails to match everything) ---
    auto t_grid_start = chrono::high_resolution_clock::now();

    // Group Red tuples into their grid cells
    unordered_map<CellKey, vector<RowId>, CellKeyHash> red_grid;
    red_grid.reserve(R * 2 + 1);
    for (size_t r = 0; r < R; ++r) {
        CellKey ck = cell_of_tuple(store.red_tuple(r), store.N, fg, threshold);
        red_grid[ck].push_back((RowId)r);
    }

    // Group Blue tuples into their grid cells
    unordered_map<CellKey, vector<RowId>, CellKeyHash> blue_grid;
    blue_grid.reserve(B * 2 + 1);
    for (size_t b = 0; b < B; ++b) {
        CellKey ck = cell_of_tuple(store.blue_tuple(b), store.N, fg, threshold);
        blue_grid[ck].push_back((RowId)b);
    }

    // Safely collect pointers to the Blue cell row vectors during sequential setup
    // This avoids concurrent lookups on blue_grid inside the threads.
    vector<pair<CellKey, const vector<RowId>*>> blue_cell_data;
    blue_cell_data.reserve(blue_grid.size());
    for (const auto& pair : blue_grid) {
        blue_cell_data.push_back({pair.first, &pair.second});
    }

    auto t_grid_end = chrono::high_resolution_clock::now();

    // Timer 2: Parallel Spatial Join
    auto t_edge_start = chrono::high_resolution_clock::now();

    const auto offsets = gen_neighbor_offsets(2 * store.N);

    // Dynamic thread allocation based on the number of unique occupied cells
    const unsigned T = min<unsigned>(SAFE_NUM_THREADS, (unsigned)max<size_t>(1, blue_cell_data.size()));
    struct LocalEdges { vector<CandidateEdge> e; };
    vector<LocalEdges> locals(T);
    vector<thread> workers;
    
    atomic<size_t> next_cell_idx(0);
    const size_t num_blue_cells = blue_cell_data.size();

    // Parallel multi-threaded worker executing the cell-to-cell spatial join safely
    auto worker = [&](unsigned tid) {
        auto& out = locals[tid].e;
        while (true) {
            size_t c_idx = next_cell_idx.fetch_add(1, memory_order_relaxed);
            if (c_idx >= num_blue_cells) break;
            
            const CellKey& bc = blue_cell_data[c_idx].first;
            const auto& b_list = *(blue_cell_data[c_idx].second);
            
            for (const auto& off : offsets) {
                CellKey nc = shifted_key(bc, off);
                auto it = red_grid.find(nc);
                if (it == red_grid.end()) continue;
                
                const auto& r_list = it->second;
                for (RowId b : b_list) {
                    const RowId* bt = store.blue_tuple(b);
                    for (RowId r : r_list) {
                        // Utilising fast cached pointer array
                        if (dist_sq_rows(bt, store.red_tuple(r), store.N, fg_ptr, th2) <= th2) {
                            out.push_back({r, b});
                        }
                    }
                }
            }
        }
    };

    for (unsigned t = 0; t < T; ++t) workers.emplace_back(worker, t);
    for (auto& th : workers) th.join();

    size_t edge_total = 0;
    for (const auto& le : locals) edge_total += le.e.size();

    auto t_edge_end = chrono::high_resolution_clock::now();

    // Timer 3: LEMON Graph Build
    auto t_graph_start = chrono::high_resolution_clock::now();

    // SmartDigraph provides a static, contiguous array-backed layout that builds
    // and stores hundreds of millions of edges with a fraction of ListDigraph's RAM.
    SmartDigraph g;
    g.reserveNode(R + B + 2);
    g.reserveArc(R + B + edge_total);
    
    SmartDigraph::ArcMap<Count> cap(g);
    vector<SmartDigraph::Node> r_nodes(R), b_nodes(B);
    SmartDigraph::Node src = g.addNode(), snk = g.addNode();
    for (size_t r = 0; r < R; ++r) { r_nodes[r] = g.addNode(); cap[g.addArc(src, r_nodes[r])] = 1; }
    
    // Set sink capacity to the exact Red bounds as safely permitted.
    for (size_t b = 0; b < B; ++b) { b_nodes[b] = g.addNode(); cap[g.addArc(b_nodes[b], snk)] = infcap; }

    vector<SmartDigraph::Arc> rb_arcs;
    vector<pair<RowId, RowId>> edge_idx;
    rb_arcs.reserve(edge_total);
    edge_idx.reserve(edge_total);

    // Build the graph. Outgoing capacity on bipartite matching arcs is set to 1.
    for (const auto& le : locals) {
        for (const auto& e : le.e) {
            auto a = g.addArc(r_nodes[e.r], b_nodes[e.b]);
            cap[a] = 1; 
            rb_arcs.push_back(a);
            edge_idx.push_back({e.r, e.b});
        }
    }
    auto t_graph_end = chrono::high_resolution_clock::now();

    // Timer 4: Preflow MaxFlow Solver
    auto t_flow_start = chrono::high_resolution_clock::now();
    Preflow<SmartDigraph, SmartDigraph::ArcMap<Count>> pf(g, cap, src, snk);
    pf.run();
    sol.flow_value = pf.flowValue();
    sol.feasible = (sol.flow_value == sol.total_red);
    auto t_flow_end = chrono::high_resolution_clock::now();

    if (!sol.feasible) {
        // Output detailed step-by-step metrics before exiting
        long long d_greedy = chrono::duration_cast<chrono::milliseconds>(t_greedy_end - t_greedy_start).count();
        long long d_grid = chrono::duration_cast<chrono::milliseconds>(t_grid_end - t_grid_start).count();
        long long d_edge = chrono::duration_cast<chrono::milliseconds>(t_edge_end - t_edge_start).count();
        long long d_graph = chrono::duration_cast<chrono::milliseconds>(t_graph_end - t_graph_start).count();
        long long d_flow = chrono::duration_cast<chrono::milliseconds>(t_flow_end - t_flow_start).count();
        cout << "  [solve_at_r=" << threshold << "] INFEASIBLE | "
             << "Greedy Matching: " << d_greedy << "ms (matches: " << greedy_matches << ") | "
             << "Grid: " << d_grid << "ms | "
             << "Parallel Join: " << d_edge << "ms (Edges: " << edge_total << ") | "
             << "LEMON Graph Build: " << d_graph << "ms | "
             << "MaxFlow Solver: " << d_flow << "ms" << endl;
        return sol;
    }

    // Timer 5: Solution Reconstruction
    auto t_recon_start = chrono::high_resolution_clock::now();
    vector<vector<pair<RowId, Count>>> incoming(B);
    for (size_t e = 0; e < rb_arcs.size(); ++e) {
        Count f = pf.flow(rb_arcs[e]);
        if (f <= 0) continue;
        incoming[edge_idx[e].second].push_back({edge_idx[e].first, f});
    }

    sol.fairlets.reserve(B);
    for (size_t b = 0; b < B; ++b) {
        int fid = (int)sol.fairlet_blue_idx.size();
        sol.blue_to_fairlet[b] = fid;
        sol.fairlet_blue_idx.push_back((RowId)b);
        
        CompactFairlet fl;
        fl.blue_idx = (RowId)b;
        fl.red_flow = std::move(incoming[b]);
        fl.rep = materialize_tuple_point(store.blue_tuple(b), store.N, fg);
        for (auto [r, f] : fl.red_flow) {
            sol.red_to_fairlets[r].push_back({fid, f});
            fl.f_r[r] = f;
        }
        sol.fairlets.push_back(std::move(fl));
    }

    sol.centers = run_gonzales_on_compact_fairlet_reps(store, fg, sol.fairlet_blue_idx, k_final);
    auto t_recon_end = chrono::high_resolution_clock::now();

    // Output detailed step-by-step metrics for successful rounds
    long long d_greedy = chrono::duration_cast<chrono::milliseconds>(t_greedy_end - t_greedy_start).count();
    long long d_grid = chrono::duration_cast<chrono::milliseconds>(t_grid_end - t_grid_start).count();
    long long d_edge = chrono::duration_cast<chrono::milliseconds>(t_edge_end - t_edge_start).count();
    long long d_graph = chrono::duration_cast<chrono::milliseconds>(t_graph_end - t_graph_start).count();
    long long d_flow = chrono::duration_cast<chrono::milliseconds>(t_flow_end - t_flow_start).count();
    long long d_recon = chrono::duration_cast<chrono::milliseconds>(t_recon_end - t_recon_start).count();
    cout << "  [solve_at_r=" << threshold << "] FEASIBLE (Fallback) | "
             << "Greedy Matching: " << d_greedy << "ms (matches: " << greedy_matches << ") | "
             << "Grid: " << d_grid << "ms | "
             << "Parallel Join: " << d_edge << "ms (Edges: " << edge_total << ") | "
             << "LEMON Graph Build: " << d_graph << "ms | "
             << "MaxFlow Solver: " << d_flow << "ms | "
             << "Reconstruction: " << d_recon << "ms" << endl;

    return sol;
}

// conference specified exact baseline cost-measurement procedure
double compute_exact_baseline_clustering_cost(const CompactJoinStore& store,
                                              const FlatGeo& fg,
                                              const CompactFairletSolution& sol) {
    if (!sol.feasible || sol.centers.empty()) throw runtime_error("cannot compute cost from an infeasible/empty baseline solution");

    vector<CompactFairlet> fairlets_copy = sol.fairlets;
    const size_t R = store.red_count();
    const size_t B = store.blue_count();
    const int N = store.N;

    // Build a map: red tuple ID -> list of fairlet IDs it belongs to
    vector<vector<int>> red_to_fids(R);
    for (int fid = 0; fid < (int)fairlets_copy.size(); ++fid) {
        for (auto const& [r_idx, f_val] : fairlets_copy[fid].red_flow) {
            if (f_val > 0) {
                red_to_fids[r_idx].push_back(fid);
            }
        }
    }

    const unsigned T = min<unsigned>(SAFE_NUM_THREADS, (unsigned)max<size_t>(1, R + B));
    vector<double> local_max(T, 0.0);
    vector<thread> workers;
    atomic<size_t> next_blue(0), next_red(0);
    const size_t CHUNK = 1024;

    auto worker = [&](unsigned tid) {
        double mx = 0.0;
        // Process Blue join results
        while (true) {
            size_t b0 = next_blue.fetch_add(CHUNK, memory_order_relaxed);
            if (b0 >= B) break;
            size_t b1 = min(B, b0 + CHUNK);
            for (size_t b = b0; b < b1; ++b) {
                int found_fid = sol.blue_to_fairlet[b];
                if (found_fid < 0) throw runtime_error("Could not find a valid fairlet for blue representative");
                
                const Point& f_rep = fairlets_copy[found_fid].rep;
                int c_idx = closest_center_idx(f_rep, sol.centers);
                const Point& c_t = sol.centers[c_idx];
                
                double dist = euclid(materialize_tuple_point(store.blue_tuple(b), N, fg), c_t);
                if (dist > mx) mx = dist;
            }
        }
        // Process Red join results
        while (true) {
            size_t r0 = next_red.fetch_add(CHUNK, memory_order_relaxed);
            if (r0 >= R) break;
            size_t r1 = min(R, r0 + CHUNK);
            for (size_t r = r0; r < r1; ++r) {
                int found_fid = -1;
                for (int fid : red_to_fids[r]) {
                    if (fairlets_copy[fid].f_r[r] > 0) {
                        found_fid = fid;
                        fairlets_copy[fid].f_r[r]--;
                        break;
                    }
                }
                if (found_fid < 0) throw runtime_error("Could not find a valid fairlet for red representative");

                const Point& f_rep = fairlets_copy[found_fid].rep;
                int c_idx = closest_center_idx(f_rep, sol.centers);
                const Point& c_t = sol.centers[c_idx];

                double dist = euclid(materialize_tuple_point(store.red_tuple(r), N, fg), c_t);
                if (dist > mx) mx = dist;
            }
        }
        local_max[tid] = mx;
    };

    for (unsigned t = 0; t < T; ++t) workers.emplace_back(worker, t);
    for (auto& th : workers) th.join();

    double max_dist = 0.0;
    for (double x : local_max) max_dist = max(max_dist, x);
    return max_dist;
}

CompactFairletSolution run_full_baseline_threshold_search(const CompactJoinStore& store,
                                                          const FlatGeo& fg,
                                                          double initial_r,
                                                          int k_final) {
    if (initial_r <= 0) throw runtime_error("initial threshold must be positive");

    double lo_bad = 0.0;
    double hi_good = initial_r;
    
    // Create the optimised flat geometry raw pointer array once
    int N = fg.size();
    vector<const double*> fg_ptr(N);
    for (int l = 0; l < N; ++l) fg_ptr[l] = fg[l].data();
    
    cout << "  --> Seeding Exact Baseline Search with Threshold=" << initial_r << endl;
    CompactFairletSolution hi_sol = solve_full_join_baseline_at_r(store, fg_ptr, fg, hi_good, k_final);

    int guard = 0;
    while (!hi_sol.feasible && guard++ < 60) {
        lo_bad = hi_good;
        hi_good *= SEARCH_FACTOR;
        cout << "  --> Scale-Up Bracketing Round " << guard << " with Threshold=" << hi_good << "..." << endl;
        hi_sol = solve_full_join_baseline_at_r(store, fg_ptr, fg, hi_good, k_final);
    }
    if (!hi_sol.feasible) return hi_sol;

    CompactFairletSolution best = hi_sol;
    double r = hi_good / SEARCH_FACTOR;
    guard = 0;
    while (r > 0 && guard++ < 60) {
        cout << "  --> Scale-Down Bracketing Round " << guard + 1 << " with Threshold=" << r << "..." << endl;
        CompactFairletSolution cur = solve_full_join_baseline_at_r(store, fg_ptr, fg, r, k_final);
        if (cur.feasible) {
            best = std::move(cur);
            hi_good = r;
            r /= SEARCH_FACTOR;
        } else {
            lo_bad = r;
            break;
        }
    }

    int step = 1;
    const double epsilon_tol = get_epsilon_tolerance();
    
    // Justified Parameter-Driven Bisection Search
    while (hi_good - lo_bad > epsilon_tol) {
        cout << "  --> Exact Baseline Binary Search Round " << step++ << " (Bracket width: " << (hi_good - lo_bad) << " ms)..." << endl;
        double mid = (lo_bad + hi_good) * 0.5;
        CompactFairletSolution cur = solve_full_join_baseline_at_r(store, fg_ptr, fg, mid, k_final);
        if (cur.feasible) {
            best = std::move(cur);
            hi_good = mid;
        } else {
            lo_bad = mid;
        }
    }
    return best;
}

// --------------------------- MAIN ---------------------------
int main() {
    try {
        int N = 4;
        const int final_k = env_int_or_default("FINAL_K", 30);
        const int approx_rep_k = env_int_or_default("APPROX_REP_K", 30);
        const double initial_threshold = 500000.0;

        if (final_k <= 0 || approx_rep_k <= 0) {
            throw runtime_error("FINAL_K and APPROX_REP_K must be positive");
        }
        cout << "Configuration: final_k=" << final_k
             << " approx_rep_k=" << approx_rep_k << endl;

        vector<Relation> g(N);
        vector<vector<KeyPair>> ky(N);

        vector<string> files = {
            env_string_or_default("FILE_L0", "yellow_tripdata_2025-01.parquet"),
            env_string_or_default("FILE_L1", "yellow_tripdata_2025-02.parquet"),
            env_string_or_default("FILE_L2", "yellow_tripdata_2025-03.parquet"),
            env_string_or_default("FILE_L3", "yellow_tripdata_2025-04.parquet")
        };

        for (int i = 0; i < N; ++i) {
            load_parquet(files[i], g[i], ky[i]);
        }

        vector<FastKeyMap> fki(N);
        for (int i = 1; i < N; i++) {
            fki[i] = FastKeyMap(ky[i].size());
            for (size_t x = 0; x < ky[i].size(); x++) {
                fki[i].ins((i % 2 != 0 ? ky[i][x].first : ky[i][x].second), (RowId)x);
            }
        }

        FlatGeo fg(N);
        for (int i = 0; i < N; i++) {
            fg[i].reserve(g[i].size() * 2);
            for (auto& p : g[i]) { fg[i].push_back(p[0]); fg[i].push_back(p[1]); }
        }

        vector<GLayer> gr(N - 1);
        vector<RowId> matches;
        for (int l = 0; l < N - 1; l++) {
            gr[l].off.reserve(ky[l].size() + 1);
            gr[l].off.push_back(0);
            for (size_t i = 0; i < ky[l].size(); i++) {
                double kv = (l % 2 == 0 ? ky[l][i].first : ky[l][i].second);
                fki[l + 1].get(kv, matches);
                gr[l].tgt.insert(gr[l].tgt.end(), matches.begin(), matches.end());
                gr[l].off.push_back((Offset)gr[l].tgt.size());
            }
        }

        cout << "Counting full join exactly before any full-join pass..." << endl;
        ExactJoinCounts jc = count_full_join_exact(fg, gr);
        cout << "Full exact join red=" << jc.red
             << " blue=" << jc.blue
             << " total=" << jc.total << endl;

        auto t1 = chrono::high_resolution_clock::now();
        cout << "Running approx coreset construction..." << endl;
        auto agg = run_approx_workflow_weighted(fg, gr, approx_rep_k);
        cout << "Approx coreset red reps=" << agg.red_centers.size()
             << " blue reps=" << agg.blue_centers.size()
             << " red weight=" << agg.count_red
             << " blue weight=" << agg.count_blue << endl;

        cout << "Running approximate fairlet threshold search with LEMON max-flow..." << endl;
        auto approx_sol = run_weighted_threshold_search(agg.red_centers, agg.red_weights,
                                                        agg.blue_centers, agg.blue_weights,
                                                        initial_threshold, final_k);
        if (!approx_sol.feasible) {
            cout << "Approx solution infeasible. flow=" << approx_sol.flow_value
                 << " red=" << approx_sol.total_red << " blue=" << approx_sol.total_blue << endl;
        } else {
            cout << "Approx threshold=" << approx_sol.threshold
                 << " fairlets=" << approx_sol.fairlets.size()
                 << " output k=" << approx_sol.centers.size() << endl;
            const Count exact_cost_limit = env_count_or_default("EXACT_COST_TUPLE_LIMIT", DEFAULT_EXACT_BASELINE_TUPLE_LIMIT);
            const bool run_exact_cost = env_flag_is_one("RUN_EXACT_COST");
            if (run_exact_cost || jc.total <= exact_cost_limit) {
                cout << "Computing approximate final measured k-center cost by exact join scan..." << endl;
                double approx_cost = compute_exact_clustering_cost(fg, gr, agg, approx_sol);
                cout << "Approx final measured k-center cost=" << setprecision(12) << approx_cost << endl;
            } else {
                cout << "Skipping approximate final exact cost scan: this query has " << jc.total
                     << " exact join tuples, above EXACT_COST_TUPLE_LIMIT=" << exact_cost_limit << ". "
                     << "Set RUN_EXACT_COST=1 to force the literal join-tuple scan." << endl;
            }
        }
        auto t2 = chrono::high_resolution_clock::now();
        cout << "Fair Approx Total Time: " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "ms" << endl;

        auto t3 = chrono::high_resolution_clock::now();
        const bool run_exact_baseline = env_flag_is_one("RUN_EXACT_BASELINE");
        const Count exact_baseline_limit = env_count_or_default("EXACT_BASELINE_TUPLE_LIMIT", DEFAULT_EXACT_BASELINE_TUPLE_LIMIT);
        if (!run_exact_baseline && jc.total > exact_baseline_limit) {
            cout << "Skipping exact baseline: this query has " << jc.total
                 << " exact join tuples, above EXACT_BASELINE_TUPLE_LIMIT=" << exact_baseline_limit << ". "
                 << "Shrink the query with QUERY_START/QUERY_END/QUERY_WINDOWS/QUERY_ORIGIN/QUERY_DEST, "
                 << "or force with RUN_EXACT_BASELINE=1." << endl;
        } else {
            cout << "Enumerating full join exactly for baseline, compact row-id storage only..." << endl;
            CompactJoinStore full = enumerate_full_join_compact(fg, gr);
            cout << "Full exact join red=" << full.red_count()
                 << " blue=" << full.blue_count()
                 << " total=" << (full.red_count() + full.blue_count()) << endl;

            cout << "Running exact full-join baseline threshold search with LEMON max-flow..." << endl;
            
            double base_initial_threshold = (approx_sol.feasible && approx_sol.threshold > 0) ? approx_sol.threshold : initial_threshold;
            
            CompactFairletSolution base_sol = run_full_baseline_threshold_search(full, fg, base_initial_threshold, final_k);
            if (!base_sol.feasible) {
                cout << "Baseline solution infeasible. flow=" << base_sol.flow_value
                     << " red=" << base_sol.total_red << " blue=" << base_sol.total_blue << endl;
            } else {
                double base_cost = compute_exact_baseline_clustering_cost(full, fg, base_sol);
                cout << "Baseline threshold=" << base_sol.threshold
                     << " fairlets=" << base_sol.fairlets.size()
                     << " output k=" << base_sol.centers.size()
                     << " final measured k-center cost=" << setprecision(12) << base_cost << endl;
            }
        }
        auto t4 = chrono::high_resolution_clock::now();
        cout << "Fair Baseline Total Time: " << chrono::duration_cast<chrono::milliseconds>(t4 - t3).count() << "ms" << endl;

    } catch (const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }
    return 0;
}