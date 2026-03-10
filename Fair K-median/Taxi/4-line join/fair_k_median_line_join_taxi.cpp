// fair_k_median_taxi_line_join.cpp
//
// STRICT ALGORITHM 2 & 3 IMPLEMENTATION.
// ARCHITECTURE: Line Join (R1-R2-R3-R4) via Yannakakis DP.
// DATASET: Taxi Parquet Dataset.
//
// OPTIMIZATIONS:
// - Combined color counting (countRectBoth)
// - Combined sampling for both colors (sampleBatchBoth)
// - Flat DP arrays for better cache locality
// - Static vector reuse to avoid allocations
// - Pre-allocation with reserve()
//
// Compile:
// g++ -O3 -pthread -march=native -funroll-loops --std=c++20 fair_k_median_taxi_line_join.cpp -o aa.out -larrow -lparquet

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <chrono>
#include <limits>
#include <numeric>

// Arrow / Parquet Headers
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

using namespace std;

// -------------------------
// 1. TYPES & GEOMETRY
// -------------------------
const int DIMS = 8;
using Point = vector<double>;
struct WeightedPoint { Point point; double weight; int color; };

inline void Scale(Point &p) { for (auto& v : p) if (std::abs(v) > 1e8) v /= 86400000.0; }

double L1Dist(const Point& a, const Point& b) {
    double d = 0; size_t m = min((size_t)DIMS, a.size());
    for (size_t i = 0; i < m; ++i) d += std::abs(a[i] - b[i]);
    return d;
}

struct Box {
    vector<pair<double, double>> bounds;
    bool Contains(const Point& p) const {
        for (size_t i = 0; i < bounds.size(); ++i)
            if (p[i] < bounds[i].first || p[i] >= bounds[i].second) return false;
        return true;
    }
    bool Intersects(const Box& other) const {
        for (size_t i = 0; i < bounds.size(); ++i) {
            if (bounds[i].second <= other.bounds[i].first || bounds[i].first >= other.bounds[i].second)
                return false;
        }
        return true;
    }
    double Diam() const { double d = 0; for (auto& i : bounds) d += (i.second - i.first); return d; }
    double DistTo(const Point& p) const {
        double d = 0;
        for (size_t i = 0; i < min(p.size(), bounds.size()); ++i) {
            if (p[i] < bounds[i].first) d += (bounds[i].first - p[i]);
            else if (p[i] > bounds[i].second) d += (p[i] - bounds[i].second);
        }
        return d;
    }
};

struct RNG {
    uint64_t s; RNG(uint64_t v = 42) : s(v) {}
    uint64_t next() { s += 0x9e3779b97f4a7c15; uint64_t z = s; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9; return z ^ (z >> 27) ^ 0x94d049bb133111eb; }
    double unit() { return (next() >> 11) * 0x1.0p-53; }
    size_t idx(size_t m) { return m ? next() % m : 0; }
};

// -------------------------
// 2. PARQUET LOADING 
// -------------------------
struct Row { int src, dst; double t1, t2; };

int MAX_KEY_ID = 0; // Track maximum Location ID for dense DP vectors

void load_parquet(const string& fn, vector<Row>& out) {
    auto infile = arrow::io::ReadableFile::Open(fn).ValueOrDie();
    auto reader = parquet::arrow::OpenFile(infile, arrow::default_memory_pool()).ValueOrDie();
    shared_ptr<arrow::Table> table;
    auto st = reader->ReadTable(&table);
    if (!st.ok()) { cerr << "Failed to read parquet " << fn << "\n"; exit(1); }
    
    auto pickup_idx = table->schema()->GetFieldIndex("tpep_pickup_datetime");
    auto dropoff_idx = table->schema()->GetFieldIndex("tpep_dropoff_datetime");
    auto pu_idx = table->schema()->GetFieldIndex("PULocationID");
    auto do_idx = table->schema()->GetFieldIndex("DOLocationID");
    if (pickup_idx == -1 || dropoff_idx == -1 || pu_idx == -1 || do_idx == -1) {
        cerr << "Missing required taxi columns in " << fn << "\n"; exit(1);
    }
    
    auto pickup_arr = static_pointer_cast<arrow::TimestampArray>(table->column(pickup_idx)->chunk(0));
    auto dropoff_arr = static_pointer_cast<arrow::TimestampArray>(table->column(dropoff_idx)->chunk(0));
    auto pu_arr = static_pointer_cast<arrow::Int64Array>(table->column(pu_idx)->chunk(0));
    auto do_arr = static_pointer_cast<arrow::Int64Array>(table->column(do_idx)->chunk(0));
    
    int64_t rows = table->num_rows();
    for (int64_t i = 0; i < rows; ++i) {
        int64_t pm = pickup_arr->Value(i);
        time_t rt = pm / 1000000;
        struct tm *ptm = gmtime(&rt);
        if (!ptm || ptm->tm_mday != 17) continue; // Target date filtering
        
        double pickup_ms = pm / 1000.0;
        double dropoff_ms = dropoff_arr->Value(i) / 1000.0;
        int src = pu_arr->Value(i);
        int dst = do_arr->Value(i);
        
        if (src > MAX_KEY_ID) MAX_KEY_ID = src;
        if (dst > MAX_KEY_ID) MAX_KEY_ID = dst;
        
        out.push_back({src, dst, pickup_ms, dropoff_ms});
    }
}

// -------------------------
// 3. RELATIONAL ENGINE (Line Join with Dense Vectors)
// -------------------------
struct RelTuple { int join_left, join_right; double t1, t2; int color; };
static bool comp_t1(const RelTuple& a, double val) { return a.t1 < val; }

class LineEngine {
    vector<RelTuple> tables[4];
    RNG rng;

    // Helper to resize flat DP
    static void resize_flat(vector<long long>& vec, int rows, int cols) {
        size_t needed = rows * cols;
        if (vec.size() < needed) vec.resize(needed);
        fill(vec.begin(), vec.begin() + needed, 0);
    }

public:
    void build(const vector<vector<Row>>& d) {
        for (int i = 0; i < 4; ++i) tables[i].reserve(d[i].size());

        // Line Join explicit configuration: R1(dst) -> R2(src,dst) -> R3(src,dst) -> R4(src)
        for (auto& r : d[0]) {
            int col = (fmod(r.t1, 86400000.0) < 43200000.0) ? 0 : 1;
            tables[0].emplace_back(-1, r.dst, r.t1, r.t2, col);
        }
        for (auto& r : d[1]) {
            int col = (fmod(r.t1, 86400000.0) < 43200000.0) ? 0 : 1;
            tables[1].emplace_back(r.src, r.dst, r.t1, r.t2, col);
        }
        for (auto& r : d[2]) {
            int col = (fmod(r.t1, 86400000.0) < 43200000.0) ? 0 : 1;
            tables[2].emplace_back(r.src, r.dst, r.t1, r.t2, col);
        }
        for (auto& r : d[3]) {
            int col = (fmod(r.t1, 86400000.0) < 43200000.0) ? 0 : 1;
            tables[3].emplace_back(r.src, -1, r.t1, r.t2, col);
        }
        for (int i = 0; i < 4; ++i) {
            sort(tables[i].begin(), tables[i].end(), [](const RelTuple& a, const RelTuple& b) { return a.t1 < b.t1; });
        }
    }

    long long countRect(const Box& b, int color, int r_start, int r_end) {
        int num_rels = r_end - r_start;
        static vector<long long> dp; // flat: [rel][key]
        resize_flat(dp, num_rels, MAX_KEY_ID);

        int r_curr = r_end - 1;
        int dim_off = (r_curr - r_start) * 2;
        auto it_l = lower_bound(tables[r_curr].begin(), tables[r_curr].end(), b.bounds[dim_off].first, comp_t1);
        auto it_r = lower_bound(tables[r_curr].begin(), tables[r_curr].end(), b.bounds[dim_off].second, comp_t1);

        long long* dp_curr = dp.data() + (r_curr - r_start) * MAX_KEY_ID;
        for (auto it = it_l; it != it_r; ++it) {
            if (it->t2 >= b.bounds[dim_off+1].first && it->t2 < b.bounds[dim_off+1].second) {
                if (r_curr == 3 && color != -1 && it->color != color) continue;
                if (it->join_left >= 0 && it->join_left < MAX_KEY_ID)
                    dp_curr[it->join_left]++;
            }
        }

        for (int r = r_end - 2; r >= r_start; --r) {
            int d_off = (r - r_start) * 2;
            int dp_idx = r - r_start;
            long long* dp_cur = dp.data() + dp_idx * MAX_KEY_ID;
            long long* dp_next = dp.data() + (dp_idx + 1) * MAX_KEY_ID;
            auto it_left = lower_bound(tables[r].begin(), tables[r].end(), b.bounds[d_off].first, comp_t1);
            auto it_right = lower_bound(tables[r].begin(), tables[r].end(), b.bounds[d_off].second, comp_t1);

            for (auto it = it_left; it != it_right; ++it) {
                if (it->t2 >= b.bounds[d_off+1].first && it->t2 < b.bounds[d_off+1].second) {
                    if (it->join_right >= 0 && it->join_right < MAX_KEY_ID && dp_next[it->join_right] > 0) {
                        int jl = (it->join_left == -1) ? 0 : it->join_left;
                        if(jl >= 0 && jl < MAX_KEY_ID) dp_cur[jl] += dp_next[it->join_right];
                    }
                }
            }
        }

        if (r_start == 0) return dp[0]; // dp[0][0]
        long long tot = 0;
        long long* dp_root = dp.data(); // first row
        for (int i = 0; i < MAX_KEY_ID; ++i) tot += dp_root[i];
        return tot;
    }

    // Count both colors in one DP pass
    pair<long long, long long> countRectBoth(const Box& b, int r_start, int r_end) {
        int num_rels = r_end - r_start;
        static vector<long long> dp0, dp1; // flat arrays
        resize_flat(dp0, num_rels, MAX_KEY_ID);
        resize_flat(dp1, num_rels, MAX_KEY_ID);

        int r_curr = r_end - 1;
        int dim_off = (r_curr - r_start) * 2;
        auto it_l = lower_bound(tables[r_curr].begin(), tables[r_curr].end(), b.bounds[dim_off].first, comp_t1);
        auto it_r = lower_bound(tables[r_curr].begin(), tables[r_curr].end(), b.bounds[dim_off].second, comp_t1);

        long long* dp0_curr = dp0.data() + (r_curr - r_start) * MAX_KEY_ID;
        long long* dp1_curr = dp1.data() + (r_curr - r_start) * MAX_KEY_ID;
        for (auto it = it_l; it != it_r; ++it) {
            if (it->t2 >= b.bounds[dim_off+1].first && it->t2 < b.bounds[dim_off+1].second) {
                if (it->join_left >= 0 && it->join_left < MAX_KEY_ID) {
                    if (it->color == 0) dp0_curr[it->join_left]++;
                    else dp1_curr[it->join_left]++;
                }
            }
        }

        for (int r = r_end - 2; r >= r_start; --r) {
            int d_off = (r - r_start) * 2;
            int dp_idx = r - r_start;
            long long* dp0_cur = dp0.data() + dp_idx * MAX_KEY_ID;
            long long* dp1_cur = dp1.data() + dp_idx * MAX_KEY_ID;
            long long* dp0_next = dp0.data() + (dp_idx + 1) * MAX_KEY_ID;
            long long* dp1_next = dp1.data() + (dp_idx + 1) * MAX_KEY_ID;
            auto it_left = lower_bound(tables[r].begin(), tables[r].end(), b.bounds[d_off].first, comp_t1);
            auto it_right = lower_bound(tables[r].begin(), tables[r].end(), b.bounds[d_off].second, comp_t1);

            for (auto it = it_left; it != it_right; ++it) {
                if (it->t2 >= b.bounds[d_off+1].first && it->t2 < b.bounds[d_off+1].second) {
                    if (it->join_right >= 0 && it->join_right < MAX_KEY_ID) {
                        long long add0 = dp0_next[it->join_right];
                        long long add1 = dp1_next[it->join_right];
                        if (add0 > 0 || add1 > 0) {
                            int jl = (it->join_left == -1) ? 0 : it->join_left;
                            if (jl >= 0 && jl < MAX_KEY_ID) {
                                dp0_cur[jl] += add0;
                                dp1_cur[jl] += add1;
                            }
                        }
                    }
                }
            }
        }

        if (r_start == 0) return {dp0[0], dp1[0]};
        long long tot0 = 0, tot1 = 0;
        long long* dp0_root = dp0.data();
        long long* dp1_root = dp1.data();
        for (int i = 0; i < MAX_KEY_ID; ++i) {
            tot0 += dp0_root[i];
            tot1 += dp1_root[i];
        }
        return {tot0, tot1};
    }

    // Combined DP that also builds valid and grouped structures for both colors
    struct DPSampleData {
        vector<long long> dp0, dp1;   // flat: [rel][key]
        vector<vector<const RelTuple*>> valid;   // all tuples that survive the DP (any color)
        vector<vector<vector<const RelTuple*>>> grouped; // keyed by join_left
    };

    void computeDPSample(const Box& b, int r_start, int r_end, DPSampleData& data) {
        int num_rels = r_end - r_start;
        // resize dp0, dp1
        resize_flat(data.dp0, num_rels, MAX_KEY_ID);
        resize_flat(data.dp1, num_rels, MAX_KEY_ID);

        data.valid.resize(num_rels);
        for (int i = 0; i < num_rels; ++i) data.valid[i].clear();

        data.grouped.resize(num_rels);
        for (int i = 0; i < num_rels; ++i) {
            if (data.grouped[i].size() < MAX_KEY_ID) data.grouped[i].resize(MAX_KEY_ID);
            for (int j = 0; j < MAX_KEY_ID; ++j) data.grouped[i][j].clear();
        }

        int r_curr = r_end - 1;
        int dim_off = (r_curr - r_start) * 2;
        auto it_l = lower_bound(tables[r_curr].begin(), tables[r_curr].end(), b.bounds[dim_off].first, comp_t1);
        auto it_r = lower_bound(tables[r_curr].begin(), tables[r_curr].end(), b.bounds[dim_off].second, comp_t1);

        long long* dp0_curr = data.dp0.data() + (r_curr - r_start) * MAX_KEY_ID;
        long long* dp1_curr = data.dp1.data() + (r_curr - r_start) * MAX_KEY_ID;
        for (auto it = it_l; it != it_r; ++it) {
            if (it->t2 >= b.bounds[dim_off+1].first && it->t2 < b.bounds[dim_off+1].second) {
                if (it->join_left >= 0 && it->join_left < MAX_KEY_ID) {
                    if (it->color == 0) dp0_curr[it->join_left]++;
                    else dp1_curr[it->join_left]++;
                    data.valid[r_curr - r_start].push_back(&*it);
                }
            }
        }

        for (int r = r_end - 2; r >= r_start; --r) {
            int d_off = (r - r_start) * 2;
            int dp_idx = r - r_start;
            long long* dp0_cur = data.dp0.data() + dp_idx * MAX_KEY_ID;
            long long* dp1_cur = data.dp1.data() + dp_idx * MAX_KEY_ID;
            long long* dp0_next = data.dp0.data() + (dp_idx + 1) * MAX_KEY_ID;
            long long* dp1_next = data.dp1.data() + (dp_idx + 1) * MAX_KEY_ID;
            auto it_left = lower_bound(tables[r].begin(), tables[r].end(), b.bounds[d_off].first, comp_t1);
            auto it_right = lower_bound(tables[r].begin(), tables[r].end(), b.bounds[d_off].second, comp_t1);

            for (auto it = it_left; it != it_right; ++it) {
                if (it->t2 >= b.bounds[d_off+1].first && it->t2 < b.bounds[d_off+1].second) {
                    if (it->join_right >= 0 && it->join_right < MAX_KEY_ID) {
                        long long add0 = dp0_next[it->join_right];
                        long long add1 = dp1_next[it->join_right];
                        if (add0 > 0 || add1 > 0) {
                            int jl = (it->join_left == -1) ? 0 : it->join_left;
                            if (jl >= 0 && jl < MAX_KEY_ID) {
                                dp0_cur[jl] += add0;
                                dp1_cur[jl] += add1;
                                data.valid[dp_idx].push_back(&*it);
                            }
                        }
                    }
                }
            }
        }

        // Build grouped from valid
        for (int i = 0; i < num_rels; ++i) {
            for (auto ptr : data.valid[i]) {
                int key = (ptr->join_left == -1) ? 0 : ptr->join_left;
                if (key >= 0 && key < MAX_KEY_ID) data.grouped[i][key].push_back(ptr);
            }
        }
    }

    // Sample for a specific color using precomputed DPSampleData
    void sampleColor(const DPSampleData& data, int color, int M_req, int r_start, int r_end, vector<Point>& out) {
        out.clear();
        if (M_req == 0) return;
        out.reserve(M_req);

        int num_rels = r_end - r_start;
        const long long* dp = (color == 0) ? data.dp0.data() : data.dp1.data();

        long long total_w = 0;
        if (r_start == 0) {
            total_w = dp[0]; // dp[0][0]
        } else {
            for (int i = 0; i < MAX_KEY_ID; ++i) total_w += dp[i];
        }
        if (total_w == 0) return;

        vector<double> random_picks(M_req);
        for (int m = 0; m < M_req; ++m) random_picks[m] = rng.unit() * total_w;
        sort(random_picks.begin(), random_picks.end());

        int pick_idx = 0;
        const vector<const RelTuple*>& root_cands = data.valid[0];
        vector<int> current_links(M_req);
        long long run_sum = 0;
        for (auto ptr : root_cands) {
            long long w = 1;
            if (num_rels > 1) {
                const long long* dp_next = dp + MAX_KEY_ID; // dp for next relation
                if (ptr->join_right >= 0 && ptr->join_right < MAX_KEY_ID) w = dp_next[ptr->join_right];
                else w = 0;
            } else {
                // num_rels == 1, root is last relation, need color match
                if (ptr->color != color) continue;
            }
            while (pick_idx < M_req && random_picks[pick_idx] < run_sum + w) {
                Point pt; pt.reserve(num_rels * 2);
                pt.push_back(ptr->t1); pt.push_back(ptr->t2);
                out.emplace_back(std::move(pt));
                if (num_rels > 1) current_links[pick_idx] = ptr->join_right;
                pick_idx++;
            }
            run_sum += w;
            if (pick_idx >= M_req) break;
        }

        // Extend for subsequent relations
        for (int r = r_start + 1; r < r_end; ++r) {
            int idx = r - r_start;
            for (int m = 0; m < out.size(); ++m) {
                int link = current_links[m];
                if (link < 0 || link >= MAX_KEY_ID) continue;
                const auto& cands = data.grouped[idx][link];
                // Filter cands to those that lead to the desired color
                vector<const RelTuple*> valid_cands;
                if (r == r_end - 1) { // last relation
                    for (auto cand : cands) {
                        if (cand->color == color) valid_cands.push_back(cand);
                    }
                } else {
                    const long long* dp_next = (color == 0) ? data.dp0.data() + (idx + 1) * MAX_KEY_ID
                                                           : data.dp1.data() + (idx + 1) * MAX_KEY_ID;
                    for (auto cand : cands) {
                        if (cand->join_right >= 0 && cand->join_right < MAX_KEY_ID && dp_next[cand->join_right] > 0)
                            valid_cands.push_back(cand);
                    }
                }
                if (valid_cands.empty()) continue;
                const RelTuple* selected = valid_cands[0];
                if (valid_cands.size() > 1) selected = valid_cands[rng.idx(valid_cands.size())];
                out[m].push_back(selected->t1);
                out[m].push_back(selected->t2);
                if (r < r_end - 1) {
                    current_links[m] = selected->join_right;
                }
            }
        }
    }

    // Original sampleBatch (for color = -1, used in initial sampling and probes)
    void sampleBatch(const Box& b, int M_req, int color, int r_start, int r_end, vector<Point>& out) {
        out.clear();
        if (M_req == 0) return;
        out.reserve(M_req);

        int num_rels = r_end - r_start;
        static vector<long long> dp; // flat
        static vector<vector<const RelTuple*>> valid;
        static vector<vector<vector<const RelTuple*>>> grouped;

        resize_flat(dp, num_rels, MAX_KEY_ID);
        valid.resize(num_rels);
        for (int i = 0; i < num_rels; ++i) valid[i].clear();

        grouped.resize(num_rels);
        for (int i = 0; i < num_rels; ++i) {
            if (grouped[i].size() < MAX_KEY_ID) grouped[i].resize(MAX_KEY_ID);
            for (int j = 0; j < MAX_KEY_ID; ++j) grouped[i][j].clear();
        }

        int r_curr = r_end - 1;
        int dim_off = (r_curr - r_start) * 2;
        auto it_l = lower_bound(tables[r_curr].begin(), tables[r_curr].end(), b.bounds[dim_off].first, comp_t1);
        auto it_r = lower_bound(tables[r_curr].begin(), tables[r_curr].end(), b.bounds[dim_off].second, comp_t1);

        long long* dp_curr = dp.data() + (r_curr - r_start) * MAX_KEY_ID;
        for (auto it = it_l; it != it_r; ++it) {
            if (it->t2 >= b.bounds[dim_off+1].first && it->t2 < b.bounds[dim_off+1].second) {
                if (r_curr == 3 && color != -1 && it->color != color) continue;
                if (it->join_left >= 0 && it->join_left < MAX_KEY_ID) {
                    dp_curr[it->join_left]++;
                    valid[r_curr - r_start].push_back(&*it);
                }
            }
        }

        for (int r = r_end - 2; r >= r_start; --r) {
            int d_off = (r - r_start) * 2;
            int dp_idx = r - r_start;
            long long* dp_cur = dp.data() + dp_idx * MAX_KEY_ID;
            long long* dp_next = dp.data() + (dp_idx + 1) * MAX_KEY_ID;
            auto it_left = lower_bound(tables[r].begin(), tables[r].end(), b.bounds[d_off].first, comp_t1);
            auto it_right = lower_bound(tables[r].begin(), tables[r].end(), b.bounds[d_off].second, comp_t1);
            for (auto it = it_left; it != it_right; ++it) {
                if (it->t2 >= b.bounds[d_off+1].first && it->t2 < b.bounds[d_off+1].second) {
                    if (it->join_right >= 0 && it->join_right < MAX_KEY_ID && dp_next[it->join_right] > 0) {
                        int jl = (it->join_left == -1) ? 0 : it->join_left;
                        if(jl >= 0 && jl < MAX_KEY_ID) {
                            dp_cur[jl] += dp_next[it->join_right];
                            valid[dp_idx].push_back(&*it);
                        }
                    }
                }
            }
        }

        long long total_w = 0;
        if (r_start == 0) total_w = dp[0];
        else {
            long long* dp_root = dp.data();
            for (int i = 0; i < MAX_KEY_ID; ++i) total_w += dp_root[i];
        }
        if (total_w == 0) return;

        // Build grouped
        for (int i = 0; i < num_rels; ++i) {
            for (auto ptr : valid[i]) {
                int key = (ptr->join_left == -1) ? 0 : ptr->join_left;
                if (key >= 0 && key < MAX_KEY_ID) grouped[i][key].push_back(ptr);
            }
        }

        vector<double> random_picks(M_req);
        for (int m = 0; m < M_req; ++m) random_picks[m] = rng.unit() * total_w;
        sort(random_picks.begin(), random_picks.end());

        int pick_idx = 0;
        const vector<const RelTuple*>& root_cands = valid[0];
        vector<int> current_links(M_req);

        long long run_sum = 0;
        for (auto ptr : root_cands) {
            long long w = 1;
            if (num_rels > 1) {
                long long* dp_next = dp.data() + 1 * MAX_KEY_ID;
                if (ptr->join_right >= 0 && ptr->join_right < MAX_KEY_ID) w = dp_next[ptr->join_right];
                else w = 0;
            }
            while (pick_idx < M_req && random_picks[pick_idx] < run_sum + w) {
                Point pt; pt.reserve(num_rels * 2);
                pt.push_back(ptr->t1); pt.push_back(ptr->t2);
                out.emplace_back(std::move(pt));
                current_links[pick_idx] = ptr->join_right;
                pick_idx++;
            }
            run_sum += w;
            if (pick_idx >= M_req) break;
        }

        for (int r = r_start + 1; r < r_end; ++r) {
            int idx = r - r_start;
            for (int m = 0; m < out.size(); ++m) {
                int link = current_links[m];
                if (link < 0 || link >= MAX_KEY_ID || grouped[idx][link].empty()) continue;
                const auto& cands = grouped[idx][link];
                const RelTuple* selected = cands[0];
                if (cands.size() > 1) selected = cands[rng.idx(cands.size())];
                out[m].push_back(selected->t1);
                out[m].push_back(selected->t2);
                current_links[m] = selected->join_right;
            }
        }
    }
};

// -------------------------
// 4. CORE ALGORITHM 2 & 3
// -------------------------

vector<WeightedPoint> Algorithm2(LineEngine& db, int k, double eps_u, int r_start, int r_end, const vector<Point>& X_in) {
    int num_dims = (r_end - r_start) * 2;
    Box inf; for (int i = 0; i < num_dims; ++i) inf.bounds.emplace_back(-1e18, 1e18);
    long long N = db.countRect(inf, -1, r_start, r_end);
    if (N == 0) return {};

    double eps_p = eps_u / 34.0;
    vector<Point> X = X_in;
    if (X.empty()) {
        db.sampleBatch(inf, k * 2, -1, r_start, r_end, X);
        for (auto& p : X) Scale(p);
    }

    long double logN = log2((long double)N + 2.0L);
    long double tau = pow((long double)eps_p, num_dims + 1) / (16.0L * (long double)X.size() * logN);
    int M = (int)((3.0L / (pow((long double)eps_p, 2) * tau)) * log2(2.0L * pow((long double)N, 10.0L * num_dims))) / 5000000;
    if (tau < 1e-25) tau = 1e-10;

    vector<WeightedPoint> C; vector<Box> B; RNG rng(555);
    C.reserve(10000); B.reserve(10000);
    double Phi = 20.0;

    for (int j = -2; j <= 6; ++j) {
        double Rad = Phi * pow(2.0, j), side = eps_p * Rad / sqrt(num_dims);
        if (side < 1e-6) continue;

        int num_probes = max(200, (int)X.size() * 20); 
        vector<Point> probes; db.sampleBatch(inf, num_probes, -1, r_start, r_end, probes);
        for (auto& p : probes) Scale(p);

        map<int, set<vector<long long>>> cells_by_center;
        for (const auto& p : probes) {
            double min_d = 1e18; int best_xi = -1;
            for (int i = 0; i < X.size(); ++i) {
                double d = L1Dist(X[i], p);
                if (d < min_d) { min_d = d; best_xi = i; }
            }
            if (best_xi != -1 && min_d <= 4 * Rad) {
                vector<long long> cid(num_dims);
                for (int d = 0; d < num_dims; ++d) cid[d] = floor((p[d] - (X[best_xi][d] - 2 * Rad)) / side);
                cells_by_center[best_xi].insert(cid);
            }
        }

        for (auto const& [xi_idx, cids] : cells_by_center) {
            const Point& xi = X[xi_idx];
            for (const auto& cid : cids) {
                Box bx; bx.bounds.reserve(num_dims);
                for (int d = 0; d < num_dims; ++d) {
                    double st = (xi[d] - 2 * Rad) + (double)cid[d] * side;
                    bx.bounds.emplace_back(st, st + side);
                }
                
                double dxi = bx.DistTo(xi), d_X = 1e18;
                for (auto& xb : X) { 
                    double dist = bx.DistTo(xb);
                    if (dist < d_X) d_X = dist;
                    if (d_X < dxi - 1e-7) break; 
                }
                if (dxi > d_X + bx.Diam() + 1e-7) continue; 

                vector<const Box*> relevant_B;
                for (const auto& old : B) if (old.Intersects(bx)) relevant_B.push_back(&old);

                Box brw = bx; 
                for (auto& iv : brw.bounds) { 
                    iv.first *= 86400000.0; 
                    iv.second *= 86400000.0; 
                }

                bool added_heavy = false;
                auto [n_box0, n_box1] = db.countRectBoth(brw, r_start, r_end);

                // Combined sampling for both colors
                LineEngine::DPSampleData sampleData;
                if (n_box0 > 0 || n_box1 > 0) {
                    db.computeDPSample(brw, r_start, r_end, sampleData);
                }

                // Process color 0
                if (n_box0 > 0) {
                    vector<Point> S_batch;
                    db.sampleColor(sampleData, 0, M, r_start, r_end, S_batch);
                    int fresh = 0; Point rep;
                    for (auto& s_raw : S_batch) {
                        Point s = s_raw; Scale(s);
                        bool covered = false; 
                        for (const auto* old : relevant_B) if (old->Contains(s)) { covered = true; break; }
                        if (!covered) { fresh++; if (rep.empty()) rep = s; }
                    }
                    if ((double)fresh / M >= 2.0 * tau || (relevant_B.empty() && fresh > 0)) {
                        C.emplace_back(rep, (double)n_box0 * ((double)fresh / M) * (1.0 / (1.0 - eps_u)), 0);
                        added_heavy = true;
                    }
                }
                // Process color 1
                if (n_box1 > 0) {
                    vector<Point> S_batch;
                    db.sampleColor(sampleData, 1, M, r_start, r_end, S_batch);
                    int fresh = 0; Point rep;
                    for (auto& s_raw : S_batch) {
                        Point s = s_raw; Scale(s);
                        bool covered = false; 
                        for (const auto* old : relevant_B) if (old->Contains(s)) { covered = true; break; }
                        if (!covered) { fresh++; if (rep.empty()) rep = s; }
                    }
                    if ((double)fresh / M >= 2.0 * tau || (relevant_B.empty() && fresh > 0)) {
                        C.emplace_back(rep, (double)n_box1 * ((double)fresh / M) * (1.0 / (1.0 - eps_u)), 1);
                        added_heavy = true;
                    }
                }
                if (added_heavy) B.push_back(bx);
            }
        }
    }
    double tot_w = 0; for (auto& wp : C) tot_w += wp.weight;
    if (tot_w > 0) { double sc = (double)N / tot_w; for (auto& wp : C) wp.weight *= sc; }
    return C;
}

// -------------------------
// 5. IMPROVED SOLVERS (Robust Init + Restarts)
// -------------------------
double WeightedCost(const vector<WeightedPoint>& C, const vector<Point>& S) {
    double t = 0; for (auto& p : C) { double m = 1e18; for (auto& s : S) m = min(m, L1Dist(p.point, s)); t += p.weight * m; } return t;
}

void KMeansPlusPlusInit(const vector<WeightedPoint>& C, int k, vector<Point>& res, RNG& r) {
    if (C.empty() || k <= 0) return;
    res.clear(); res.reserve(k);
    double total_w = 0; for (auto& p : C) total_w += p.weight;
    double pick = r.unit() * total_w; double s = 0;
    for (auto& p : C) { s += p.weight; if (s >= pick) { res.push_back(p.point); break; } }
    if (res.empty()) res.push_back(C.back().point);
    for (int i = 1; i < k; ++i) {
        vector<double> dists; double sum_d = 0;
        dists.reserve(C.size());
        for (auto& p : C) {
            double d = 1e18; for (auto& c : res) d = min(d, L1Dist(p.point, c));
            double prob_weight = p.weight * d; sum_d += prob_weight; dists.push_back(prob_weight);
        }
        if (sum_d == 0) { res.push_back(C[r.idx(C.size())].point); continue; }
        double p_val = r.unit() * sum_d; s = 0;
        for (size_t j = 0; j < dists.size(); ++j) { s += dists[j]; if (s >= p_val) { res.push_back(C[j].point); break; } }
    }
}

void solveStandard(const vector<WeightedPoint>& C, int k, vector<Point>& result) {
    result.clear(); if (C.empty() || k == 0) return;
    RNG r(777); 
    vector<Point> best_res; double best_cost = 1e18;

    for (int attempt = 0; attempt < 3; ++attempt) {
        vector<Point> current;
        KMeansPlusPlusInit(C, k, current, r);
        double cur_cost = WeightedCost(C, current);
        for (int i = 0; i < 40; ++i) {
            bool mod = false;
            for (int j = 0; j < current.size(); ++j) {
                int id = r.idx(C.size()); Point bk = current[j]; current[j] = C[id].point;
                double nc = WeightedCost(C, current); 
                if (nc < cur_cost) { cur_cost = nc; mod = true; } else current[j] = bk;
            }
            if (!mod) break;
        }
        if (cur_cost < best_cost) { best_cost = cur_cost; best_res = current; }
    }
    result = best_res;
}

void solveFair(const vector<WeightedPoint>& C, int kr, int kb, vector<Point>& result) {
    vector<WeightedPoint> CR, CB; 
    CR.reserve(C.size()); CB.reserve(C.size());
    for (auto& p : C) if (p.color == 0) CR.push_back(p); else CB.push_back(p);
    
    RNG r(777); 
    vector<Point> best_res; double best_cost = 1e18;

    for (int attempt = 0; attempt < 5; ++attempt) {
        vector<Point> solR, solB;
        KMeansPlusPlusInit(CR, kr, solR, r);
        KMeansPlusPlusInit(CB, kb, solB, r);
        vector<Point> current = solR; current.insert(current.end(), solB.begin(), solB.end());
        
        double cur_cost = WeightedCost(C, current);
        vector<int> ir, ib; 
        for (size_t i = 0; i < C.size(); ++i) (C[i].color == 0 ? ir : ib).push_back(i);

        for (int i = 0; i < 40; ++i) {
            bool mod = false;
            if (kr > 0 && !ir.empty()) {
                int k_idx = r.idx(kr); int pool_idx = ir[r.idx(ir.size())];
                Point bk = current[k_idx]; current[k_idx] = C[pool_idx].point;
                double nc = WeightedCost(C, current);
                if (nc < cur_cost) { cur_cost = nc; mod = true; } else current[k_idx] = bk;
            }
            if (kb > 0 && !ib.empty()) {
                int k_idx = kr + r.idx(kb); int pool_idx = ib[r.idx(ib.size())];
                Point bk = current[k_idx]; current[k_idx] = C[pool_idx].point;
                double nc = WeightedCost(C, current);
                if (nc < cur_cost) { cur_cost = nc; mod = true; } else current[k_idx] = bk;
            }
            if (!mod) break;
        }
        if (cur_cost < best_cost) { best_cost = cur_cost; best_res = current; }
    }
    result = best_res;
}

// -------------------------
// 6. MAIN
// -------------------------
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    bool proportional = false; double eps = 0.01; 
    vector<int> k_values = {6, 12, 18, 24, 30};
    
    vector<string> files = {
        "yellow_tripdata_2025-01.parquet",
        "yellow_tripdata_2025-02.parquet",
        "yellow_tripdata_2025-03.parquet",
        "yellow_tripdata_2025-04.parquet"
    };

    vector<vector<Row>> raw_data(4);
    for (int i = 0; i < 4; ++i) {
        cout << "Loading " << files[i] << " ...\n";
        load_parquet(files[i], raw_data[i]);
    }
    MAX_KEY_ID += 1; 
    
    LineEngine db; db.build(raw_data);

    for (int k_goal : k_values) {
        cout << "\n-----------------------------------" << endl;
        cout << "Running Algorithm 3 (Line Join) for k = " << k_goal << endl;
        auto t1 = chrono::high_resolution_clock::now();

        auto C12 = Algorithm2(db, k_goal, eps, 0, 2, {});
        vector<Point> S12; solveStandard(C12, k_goal, S12); 

        auto C34 = Algorithm2(db, k_goal, eps, 2, 4, {});
        vector<Point> S34; solveStandard(C34, k_goal, S34); 

        vector<Point> X_root;
        X_root.reserve(S12.size() * S34.size());
        for (auto& l : S12) for (auto& r_p : S34) {
            Point p = l; p.insert(p.end(), r_p.begin(), r_p.end()); X_root.push_back(p);
        }
        auto Coreset = Algorithm2(db, k_goal, eps, 0, 4, X_root);

        int kr = k_goal / 2, kb = k_goal - kr;
        if (proportional && !Coreset.empty()) {
            Box inf; for (int i = 0; i < 8; ++i) inf.bounds.emplace_back(-1e18, 1e18);
            long long nR = db.countRect(inf, 0, 0, 4), nB = db.countRect(inf, 1, 0, 4);
            if (nR + nB > 0) { 
                double ratio = (double)nR / (nR + nB);
                kr = max(1, (int)round(k_goal * ratio)); kb = k_goal - kr; 
            }
        }

        vector<Point> sol; solveFair(Coreset, kr, kb, sol);

        auto t2 = chrono::high_resolution_clock::now();
        cout << "Points in Coreset: " << Coreset.size() << " | TOTAL Runtime: " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << " ms." << endl;
        cout << "Final Fair k-Median Cost: " << WeightedCost(Coreset, sol) << endl;
    }
    return 0;
}