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
#include <future>
#include <atomic>
#include <random>
#include <cstdint>
#include <mutex>
#include <numeric>

using namespace std;

// --------------------------- CONFIG & TYPES ---------------------------
const unsigned SAFE_NUM_THREADS = (std::thread::hardware_concurrency() == 0) ? 1u : std::thread::hardware_concurrency();
using Point = vector<double>;
using Relation = vector<Point>;
using KeyPair = pair<double, double>;
using FlatGeo = vector<vector<double>>;

struct CoresetResult {
    vector<Point> red_centers;
    vector<Point> blue_centers;
    size_t count_red;
    size_t count_blue;
};

// --------------------------- UTILS ---------------------------
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline uint64_t dbl_to_u64(double d) {
    uint64_t u; memcpy(&u, &d, sizeof(d)); return u;
}

static inline double u64_to_dbl(uint64_t u) {
    double d; memcpy(&d, &u, sizeof(u)); return d;
}

double parseTime(const string& t) {
    if (t.size() < 19) return 0.0;
    int yy = stoi(t.substr(0, 4)), mm = stoi(t.substr(5, 2)), dd = stoi(t.substr(8, 2));
    int h = stoi(t.substr(11, 2)), m = stoi(t.substr(14, 2)), s = stoi(t.substr(17, 2));
    static const int mdays[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    auto is_leap = [](int y) { return (y % 4 == 0) && ((y % 100) != 0 || (y % 400) == 0); };
    long long days = 0; for (int y = 1900; y < yy; ++y) days += 365 + (is_leap(y) ? 1 : 0);
    for (int mo = 1; mo < mm; ++mo) { days += mdays[mo]; if (mo == 2 && is_leap(yy)) ++days; }
    days += (dd - 1);
    return (double)(days * 86400LL + h * 3600 + m * 60 + s) * 1000.0;
}

inline double euclid(const Point &a, const Point &b) {
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) { double d = a[i] - b[i]; sum += d * d; }
    return sqrt(max(0.0, sum));
}

class MapStringId {
    unordered_map<string, double> s2d; double nxt = 1.0;
public:
    double getId(const string& s) { if (s2d.find(s) == s2d.end()) s2d[s] = nxt++; return s2d[s]; }
};

struct FastKeyMap {
    vector<vector<uint32_t>> table; size_t mask;
    FastKeyMap(size_t ex = 0) {
        size_t sz = 1; while (sz <= ex) sz <<= 1;
        if (sz < 8) sz = 8; table.resize(sz); mask = sz - 1;
    }
    void ins(double k, uint32_t v) { table[splitmix64(dbl_to_u64(k)) & mask].push_back(v); }
    const vector<uint32_t>* get(double k) const {
        const auto& v = table[splitmix64(dbl_to_u64(k)) & mask];
        return v.empty() ? nullptr : &v;
    }
};

struct FastSet {
    vector<uint64_t> table; size_t mask;
    FastSet(size_t p2) { size_t sz = 1ULL << p2; table.resize(sz, 0); mask = sz - 1; }
    inline bool insert(uint64_t v) {
        if (v == 0) v = 1; size_t i = splitmix64(v) & mask;
        while (true) {
            uint64_t cur = table[i];
            if (cur == 0) { table[i] = v; return true; }
            if (cur == v) return false;
            i = (i + 1) & mask;
        }
    }
};

struct GLayer { vector<uint32_t> off; vector<uint32_t> tgt; };

static atomic<uint64_t> g_mx_bits;
static mutex mx_mtx;
static Point g_max_point;

enum class TaskType { FARTHEST, ADJACENCY, SWAP_POINT };

struct JobContext {
    int N; int K; 
    const FlatGeo* fg; const vector<GLayer>* gr; // Graph Structure Array
    const double* cx; const double* cy; const double* bnd;
    atomic<size_t>* cu; size_t tot; int tc;
    
    int K_base;
    const double* bx; const double* by;
    const bool* base_in_G;
    const double* base_bnd;

    int fromG, toG;
    const int* sexes;
    atomic<bool>* found_swap;
    atomic<bool>* all_adj_found;
    Point* swap_result;
    int* swap_center_idx;

    atomic<int>* adj_matrix; 
};

// --------------------------- STAR TOPOLOGY DFS SEARCHER ---------------------------
template<TaskType TASK>
void worker_generic(JobContext ctx) {
    int N = ctx.N, K = ctx.K; 
    alignas(64) double acc[1536]; 
    alignas(64) double b_acc[1536]; 
    alignas(32) double p_coords[64];
    
    const double* cx = ctx.cx; const double* cy = ctx.cy; const double* bptr = ctx.bnd;
    const double* bx = ctx.bx; const double* by = ctx.by;

    // Traverses paths of elements structured across a strict Star architecture 
    // branching outwardly continuously around R[0] 
    auto dfs = [&](auto&& self, int l, uint32_t root_id, uint32_t dim_id, int col) -> void {
        if constexpr (TASK == TaskType::ADJACENCY) if (ctx.all_adj_found->load(memory_order_relaxed)) return;

        if (l == 0) {
            col = (fmod((*ctx.fg)[0][2 * dim_id], 86400000.0) < 43200000.0) ? 0 : 1;
            if (ctx.tc != -1 && col != ctx.tc) return;
            if constexpr (TASK == TaskType::SWAP_POINT) { if (col != ctx.toG) return; }
            if constexpr (TASK == TaskType::ADJACENCY) {
                bool need_0 = ctx.adj_matrix[0 * 2 + col].load(memory_order_relaxed) == 0;
                bool need_1 = ctx.adj_matrix[1 * 2 + col].load(memory_order_relaxed) == 0;
                if (!need_0 && !need_1) return;
            }
        }
        
        // Grab Geometry coordinates locally
        double px = (*ctx.fg)[l][2 * dim_id], py = (*ctx.fg)[l][2 * dim_id + 1];
        p_coords[2*l] = px; p_coords[2*l+1] = py;
        
        double* cur = acc + (l * max(1, K));
        double* pre = (l > 0) ? acc + ((l - 1) * max(1, K)) : nullptr;
        double* b_cur = (ctx.K_base > 0) ? b_acc + (l * ctx.K_base) : nullptr;
        double* b_pre = (ctx.K_base > 0 && l > 0) ? b_acc + ((l - 1) * ctx.K_base) : nullptr;

        // Terminal Star Check evaluation bounds
        if (l == N - 1) {
            if constexpr (TASK == TaskType::FARTHEST) {
                if (ctx.K_base > 0) {
                    double min_b = 1e30; int best_b = 0;
                    for(int c=0; c<ctx.K_base; ++c) {
                        double dx = px - bx[l*ctx.K_base+c], dy = py - by[l*ctx.K_base+c];
                        double d = (b_pre ? b_pre[c] : 0.0) + dx*dx + dy*dy;
                        if (d < min_b) { min_b = d; best_b = c; }
                    }
                    if (!ctx.base_in_G[best_b]) return; 
                }
                
                if (K == 0) {
                    bool expected = false;
                    if (ctx.found_swap->compare_exchange_strong(expected, true)) {
                        lock_guard<mutex> lk(mx_mtx);
                        g_max_point.assign(p_coords, p_coords + 2 * N);
                    }
                    return;
                }

                double min_d = 1e30;
                double current_max = u64_to_dbl(g_mx_bits.load(memory_order_relaxed));
                bool valid = true;
                for (int c = 0; c < K; ++c) {
                    double dx = px - cx[l*K+c], dy = py - cy[l*K+c];
                    double d = (pre ? pre[c] : 0.0) + dx * dx + dy * dy;
                    if (d <= current_max) { valid = false; break; } // Mathematical O(1) Fast Break!
                    if (d < min_d) min_d = d;
                }
                
                if (valid && min_d > current_max) {
                    lock_guard<mutex> lk(mx_mtx);
                    if (min_d > u64_to_dbl(g_mx_bits.load(memory_order_relaxed))) {
                        g_mx_bits.store(dbl_to_u64(min_d), memory_order_relaxed);
                        g_max_point.assign(p_coords, p_coords + 2 * N);
                    }
                }
            }
            else if constexpr (TASK == TaskType::ADJACENCY) {
                double min_d = 1e30; int best_c = 0;
                for (int c = 0; c < K; ++c) {
                    double dx = px - cx[l*K+c], dy = py - cy[l*K+c];
                    double d = (pre ? pre[c] : 0.0) + dx * dx + dy * dy;
                    if (d < min_d) { min_d = d; best_c = c; }
                }
                int idx = ctx.sexes[best_c] * 2 + col;
                if (ctx.adj_matrix[idx].load(memory_order_relaxed) == 0) {
                    ctx.adj_matrix[idx].store(1, memory_order_relaxed);
                    bool all_done = true;
                    for(int i=0; i<4; ++i) if (ctx.adj_matrix[i].load(memory_order_relaxed) == 0) all_done = false;
                    if (all_done) ctx.all_adj_found->store(true, memory_order_relaxed);
                }
            }
            else if constexpr (TASK == TaskType::SWAP_POINT) {
                double min_d = 1e30; int best_c = 0;
                for (int c = 0; c < K; ++c) {
                    double dx = px - cx[l*K+c], dy = py - cy[l*K+c];
                    double d = (pre ? pre[c] : 0.0) + dx * dx + dy * dy;
                    if (d < min_d) { min_d = d; best_c = c; }
                }
                if (ctx.sexes[best_c] == ctx.fromG) {
                    double current_min = u64_to_dbl(g_mx_bits.load(memory_order_relaxed));
                    if (min_d < current_min) {
                        lock_guard<mutex> lk(mx_mtx);
                        if (min_d < u64_to_dbl(g_mx_bits.load(memory_order_relaxed))) {
                            g_mx_bits.store(dbl_to_u64(min_d), memory_order_relaxed);
                            ctx.swap_result->assign(p_coords, p_coords + 2 * N);
                            *ctx.swap_center_idx = best_c;
                        }
                    }
                }
            }
        } else {
            // BOUNDING EVALUATION: Propagating over Independent Topologic Structures
            if constexpr (TASK == TaskType::FARTHEST) {
                if (K > 0) {
                    bool pruned = false;
                    double current_max = u64_to_dbl(g_mx_bits.load(memory_order_relaxed));
                    for (int c = 0; c < K; ++c) {
                        double dx = px - cx[l*K+c], dy = py - cy[l*K+c];
                        double v = (pre ? pre[c] : 0.0) + dx*dx + dy*dy;
                        cur[c] = v;
                        if (bptr && (v + bptr[(l+1)*K+c] <= current_max)) { pruned = true; break; }
                    }
                    if (pruned) return; 
                }

                if (ctx.K_base > 0) {
                    double min_lb_inG = 1e30, min_ub_outG = 1e30;
                    for(int c=0; c<ctx.K_base; ++c) {
                        double dx = px - bx[l*ctx.K_base+c], dy = py - by[l*ctx.K_base+c];
                        double v = (b_pre ? b_pre[c] : 0.0) + dx*dx + dy*dy;
                        b_cur[c] = v;
                        if (ctx.base_in_G[c]) { if (v < min_lb_inG) min_lb_inG = v; } 
                        else {
                            double ub = v + (ctx.base_bnd ? ctx.base_bnd[(l+1)*ctx.K_base+c] : 0);
                            if (ub < min_ub_outG) min_ub_outG = ub;
                        }
                    }
                    if (ctx.base_bnd && min_ub_outG < min_lb_inG) return; 
                }
            }
            else if constexpr (TASK == TaskType::ADJACENCY) {
                bool need_0 = ctx.adj_matrix[0 * 2 + col].load(memory_order_relaxed) == 0;
                bool need_1 = ctx.adj_matrix[1 * 2 + col].load(memory_order_relaxed) == 0;
                if (!need_0 && !need_1) return;

                double min_lb[2] = {1e30, 1e30}, min_ub[2] = {1e30, 1e30};
                for(int c=0; c<K; ++c) {
                    double dx = px - cx[l*K+c], dy = py - cy[l*K+c];
                    double v = (pre ? pre[c] : 0.0) + dx*dx + dy*dy;
                    cur[c] = v;
                    double ub = v + (bptr ? bptr[(l+1)*K+c] : 0);
                    int s = ctx.sexes[c];
                    if (v < min_lb[s]) min_lb[s] = v;
                    if (ub < min_ub[s]) min_ub[s] = ub;
                }
                if (bptr) {
                    bool can_find_0 = need_0 && (min_lb[0] <= min_ub[1]);
                    bool can_find_1 = need_1 && (min_lb[1] <= min_ub[0]);
                    if (!can_find_0 && !can_find_1) return; 
                }
            }
            else if constexpr (TASK == TaskType::SWAP_POINT) {
                double min_lb_target = 1e30, min_ub_other = 1e30;
                for(int c=0; c<K; ++c) {
                    double dx = px - cx[l*K+c], dy = py - cy[l*K+c];
                    double v = (pre ? pre[c] : 0.0) + dx*dx + dy*dy;
                    cur[c] = v;
                    if (ctx.sexes[c] == ctx.fromG) { if (v < min_lb_target) min_lb_target = v; } 
                    else {
                        double ub = v + (bptr ? bptr[(l+1)*K+c] : 0);
                        if (ub < min_ub_other) min_ub_other = ub;
                    }
                }
                double current_min = u64_to_dbl(g_mx_bits.load(memory_order_relaxed));
                if (bptr && min_ub_other < min_lb_target) return; 
                if (min_lb_target >= current_min) return; // Strict bounds checking Optimization! 
            }

            // Cross into mapped subset array attached from strictly root_id via direct structural relationships.
            uint32_t s = (*ctx.gr)[l + 1].off[root_id];
            uint32_t e = (*ctx.gr)[l + 1].off[root_id + 1];
            for (uint32_t k = s; k < e; ++k) {
                self(self, l + 1, root_id, (*ctx.gr)[l + 1].tgt[k], col);
            }
        }
    };

    size_t BS = 512;
    while (true) {
        if constexpr (TASK == TaskType::ADJACENCY) if (ctx.all_adj_found->load(memory_order_relaxed)) break;
        if constexpr (TASK == TaskType::FARTHEST) if (K == 0 && ctx.found_swap->load(memory_order_relaxed)) break;
        
        size_t s = ctx.cu->fetch_add(BS, memory_order_relaxed);
        if (s >= ctx.tot) break;
        size_t e = min(s + BS, ctx.tot);
        // Dispatch mapping processes passing initial dataset index values cleanly into parallel paths 
        for (size_t i = s; i < e; ++i) dfs(dfs, 0, (uint32_t)i, (uint32_t)i, -1);
    }
}

// --------------------------- EXACT WORKFLOW STRUCTURE EXECUTIONS ---------------------------
vector<double> compute_bnd(int N, const FlatGeo& fg, int K_cur, const vector<double>& cx, const vector<double>& cy) {
    vector<double> fb((N + 1) * max(1, K_cur), 0.0);
    if (K_cur == 0) return fb;
    vector<vector<double>> tmp(K_cur, vector<double>(N + 1, 0.0));
    for (int l = 0; l < N; ++l) {
        size_t sz = fg[l].size() / 2; const double* pts = fg[l].data();
        vector<atomic<uint64_t>> max_d_u64(K_cur);
        for(int c=0; c<K_cur; ++c) max_d_u64[c].store(0);
        
        auto calc_bounds = [&](size_t start, size_t end) {
            vector<double> local_max(K_cur, 0.0);
            for (size_t i = start; i < end; ++i) {
                for (int c = 0; c < K_cur; ++c) {
                    double dx = pts[2 * i] - cx[l * K_cur + c], dy = pts[2 * i + 1] - cy[l * K_cur + c];
                    double d = dx * dx + dy * dy;
                    if (d > local_max[c]) local_max[c] = d;
                }
            }
            for (int c = 0; c < K_cur; ++c) {
                uint64_t loc = dbl_to_u64(local_max[c]), cur = max_d_u64[c].load(memory_order_relaxed);
                while (loc > cur && !max_d_u64[c].compare_exchange_weak(cur, loc, memory_order_relaxed)) {}
            }
        };
        vector<thread> th_b; size_t chunk = sz / SAFE_NUM_THREADS + 1;
        for(unsigned t=0; t<SAFE_NUM_THREADS; ++t){
            size_t st = min(t * chunk, sz), en = min((t+1) * chunk, sz);
            if (st < en) th_b.emplace_back(calc_bounds, st, en);
        }
        for(auto& t : th_b) t.join();
        for(int c=0; c<K_cur; ++c) tmp[c][l] = u64_to_dbl(max_d_u64[c].load());
    }
    for (int c = 0; c < K_cur; ++c) {
        for (int l = N - 1; l >= 0; --l) tmp[c][l] += tmp[c][l + 1];
        for (int l = 0; l <= N; ++l) fb[l * K_cur + c] = tmp[c][l];
    }
    return fb;
}

pair<Point, bool> findAnyJoinTuple(int N, const FlatGeo& f, const vector<GLayer>& gr, int target_color) {
    Point wk(2 * N);
    // Adjusted DFS traversing the specific star combination patterns utilizing structural relations accurately 
    function<bool(int,uint32_t,uint32_t)> dfs = [&](int l, uint32_t root_id, uint32_t dim_id) -> bool {
        if (l == 0 && target_color != -1) {
            int c = (int)(fmod(f[0][2 * dim_id], 86400000.0) < 43200000.0 ? 0 : 1);
            if (c != target_color) return false;
        }
        wk[2*l]=f[l][2*dim_id]; wk[2*l+1]=f[l][2*dim_id+1];
        if (l == N - 1) return true;
        
        uint32_t st = gr[l+1].off[root_id], en = gr[l+1].off[root_id+1];
        for (uint32_t x = st; x < en; ++x) if (dfs(l+1, root_id, gr[l+1].tgt[x])) return true;
        return false;
    };
    for (size_t i = 0; i < f[0].size()/2; ++i) if (dfs(0, i, i)) return {wk, true};
    return {wk, false};
}

vector<Point> exact_fair_k_center(int N, const FlatGeo& fg, const vector<GLayer>& gr, int kr, int kb) {
    int K_total = kr + kb;
    if (K_total == 0) return {};
    auto get_color = [](const Point& p) { return (fmod(p[0], 86400000.0) < 43200000.0) ? 0 : 1; };

    auto s_res = findAnyJoinTuple(N, fg, gr, kr > 0 ? 0 : 1);
    if (!s_res.second) return {};
    vector<Point> centersTE = {s_res.first};

    while ((int)centersTE.size() < K_total) {
        JobContext ctx; ctx.N = N; ctx.K = centersTE.size(); ctx.fg = &fg; ctx.gr = &gr;
        vector<double> cx(N * max(1, ctx.K)), cy(N * max(1, ctx.K));
        for (int l = 0; l < N; ++l) { for (int c = 0; c < ctx.K; ++c) { cx[l * ctx.K + c] = centersTE[c][2 * l]; cy[l * ctx.K + c] = centersTE[c][2 * l + 1]; } }
        ctx.cx = cx.data(); ctx.cy = cy.data();
        
        vector<double> fb = compute_bnd(N, fg, ctx.K, cx, cy);
        ctx.bnd = (ctx.K > 0) ? fb.data() : nullptr;
        ctx.K_base = 0; ctx.bx = nullptr; ctx.by = nullptr; ctx.base_in_G = nullptr; ctx.base_bnd = nullptr;
        
        atomic<size_t> cu(0); ctx.cu = &cu; ctx.tot = fg[0].size() / 2; ctx.tc = -1;
        g_mx_bits.store(dbl_to_u64(0.0)); g_max_point.clear();
        atomic<bool> fa(false); ctx.found_swap = &fa; 
        
        vector<thread> th;
        for (unsigned i = 0; i < SAFE_NUM_THREADS; ++i) th.emplace_back([&]() { worker_generic<TaskType::FARTHEST>(ctx); });
        for (auto& t : th) t.join();
        centersTE.push_back(g_max_point);
    }

    int m = 2; vector<int> nrs = {kr, kb};
    while (true) {
        vector<int> sexes(K_total); for(int c=0; c<K_total; ++c) sexes[c] = get_color(centersTE[c]);
        vector<int> curr_cnt(m, 0); for(int c=0; c<K_total; ++c) curr_cnt[sexes[c]]++;
        
        atomic<int> adj_atomic[4]; for(int i=0;i<4;++i) adj_atomic[i].store(0);
        atomic<bool> all_adj_found(false);
        
        JobContext ctx; ctx.N = N; ctx.K = K_total; ctx.fg = &fg; ctx.gr = &gr;
        vector<double> cx(N * K_total), cy(N * K_total);
        for (int l = 0; l < N; ++l) { for (int c = 0; c < K_total; ++c) { cx[l * K_total + c] = centersTE[c][2 * l]; cy[l * K_total + c] = centersTE[c][2 * l + 1]; } }
        ctx.cx = cx.data(); ctx.cy = cy.data(); ctx.sexes = sexes.data();
        
        vector<double> fb = compute_bnd(N, fg, K_total, cx, cy);
        ctx.bnd = fb.data(); ctx.adj_matrix = adj_atomic; ctx.all_adj_found = &all_adj_found;
        atomic<size_t> cu(0); ctx.cu = &cu; ctx.tot = fg[0].size() / 2; ctx.tc = -1;
        
        vector<thread> th;
        for (unsigned i = 0; i < SAFE_NUM_THREADS; ++i) th.emplace_back([&]() { worker_generic<TaskType::ADJACENCY>(ctx); });
        for (auto& t : th) t.join();
        
        int adj[2][2]; for(int i=0;i<2;++i) for(int j=0;j<2;++j) adj[i][j] = adj_atomic[i*2+j].load();
        
        const int INF = 1e9; vector<vector<int>> dist(m, vector<int>(m, INF)), pred(m, vector<int>(m, -1));
        for(int i=0;i<m;++i){ dist[i][i]=0; for(int j=0;j<m;++j){ if(adj[i][j]){ dist[i][j]=1; pred[i][j]=i; } } }
        for(int kk=0;kk<m;++kk) for(int i=0;i<m;++i) for(int j=0;j<m;++j)
            if(dist[i][kk]+dist[kk][j] < dist[i][j]){ dist[i][j] = dist[i][kk]+dist[kk][j]; pred[i][j] = pred[kk][j]; }
        
        bool found = false; vector<int> path;
        for(int l=0; l<m && !found; ++l){
            if(curr_cnt[l] > nrs[l]){
                for(int z=0; z<m; ++z){
                    if(curr_cnt[z] < nrs[z] && dist[l][z] < INF){
                        int cur=z; path.push_back(z);
                        while(cur!=l){ cur = pred[l][cur]; path.insert(path.begin(), cur); }
                        found=true; break;
                    }
                }
            }
        }
        
        if (found) {
            for(size_t h=0; h+1<path.size(); ++h){
                int fromG = path[h], toG = path[h+1];
                Point swap_result; int swap_center_idx = -1;
                
                JobContext s_ctx; s_ctx.N = N; s_ctx.K = K_total; s_ctx.fg = &fg; s_ctx.gr = &gr;
                s_ctx.cx = cx.data(); s_ctx.cy = cy.data(); s_ctx.bnd = fb.data();
                s_ctx.sexes = sexes.data(); s_ctx.fromG = fromG; s_ctx.toG = toG;
                s_ctx.swap_result = &swap_result; s_ctx.swap_center_idx = &swap_center_idx;
                atomic<size_t> scu(0); s_ctx.cu = &scu; s_ctx.tot = fg[0].size() / 2; s_ctx.tc = -1;
                atomic<bool> sc(false); s_ctx.found_swap = &sc;
                
                g_mx_bits.store(dbl_to_u64(1e30), memory_order_relaxed); 
                
                vector<thread> sth;
                for (unsigned i = 0; i < SAFE_NUM_THREADS; ++i) sth.emplace_back([&]() { worker_generic<TaskType::SWAP_POINT>(s_ctx); });
                for (auto& t : sth) t.join();
                
                if (swap_center_idx != -1) centersTE[swap_center_idx] = swap_result;
            }
        } else {
            bool ok = true; for(int i=0; i<m; ++i) if(curr_cnt[i] != nrs[i]) ok = false;
            if (ok) return centersTE;
            
            vector<int> G;
            for(int i=0;i<m;++i) if(curr_cnt[i]>nrs[i]) G.push_back(i);
            for(int i=0;i<m;++i) for(int j=0;j<m;++j) if(dist[i][j]<INF && find(G.begin(),G.end(),i)!=G.end() && find(G.begin(),G.end(),j)==G.end()) G.push_back(j);
                        
            vector<Point> new_given; vector<bool> in_G(K_total, false);
            for(int c=0; c<K_total; ++c) {
                if(find(G.begin(), G.end(), sexes[c]) != G.end()) in_G[c] = true; else new_given.push_back(centersTE[c]);
            }
            
            vector<Point> base_centers = centersTE; centersTE = new_given;
            
            vector<double> base_cx(N * max(1, K_total)), base_cy(N * max(1, K_total));
            for(int l=0; l<N; ++l){ for(int c=0; c<K_total; ++c){ base_cx[l*K_total+c] = base_centers[c][2*l]; base_cy[l*K_total+c] = base_centers[c][2*l+1]; } }
            vector<double> base_bnd = compute_bnd(N, fg, K_total, base_cx, base_cy);
            
            while((int)centersTE.size() < K_total) {
                JobContext r_ctx; r_ctx.N = N; r_ctx.K = centersTE.size(); r_ctx.fg = &fg; r_ctx.gr = &gr;
                vector<double> r_cx(N * max(1, r_ctx.K)), r_cy(N * max(1, r_ctx.K));
                for (int l = 0; l < N; ++l) { for (int c = 0; c < r_ctx.K; ++c) { r_cx[l * r_ctx.K + c] = centersTE[c][2 * l]; r_cy[l * r_ctx.K + c] = centersTE[c][2 * l + 1]; } }
                r_ctx.cx = r_cx.data(); r_ctx.cy = r_cy.data();
                
                vector<double> r_fb = compute_bnd(N, fg, r_ctx.K, r_cx, r_cy); r_ctx.bnd = (r_ctx.K > 0) ? r_fb.data() : nullptr;
                
                r_ctx.K_base = base_centers.size(); r_ctx.bx = base_cx.data(); r_ctx.by = base_cy.data();
                r_ctx.base_bnd = base_bnd.data();
                bool base_in_G_arr[64]; for(int c=0; c<r_ctx.K_base; ++c) base_in_G_arr[c] = in_G[c]; r_ctx.base_in_G = base_in_G_arr;
                
                atomic<size_t> rcu(0); r_ctx.cu = &rcu; r_ctx.tot = fg[0].size() / 2; r_ctx.tc = -1;
                g_mx_bits.store(dbl_to_u64(0.0)); g_max_point.clear();
                atomic<bool> rfa(false); r_ctx.found_swap = &rfa;
                
                vector<thread> rth;
                for (unsigned i = 0; i < SAFE_NUM_THREADS; ++i) rth.emplace_back([&]() { worker_generic<TaskType::FARTHEST>(r_ctx); });
                for (auto& t : rth) t.join();
                centersTE.push_back(g_max_point);
            }
            return centersTE; 
        }
    }
}

pair<vector<Point>, vector<Point>> solveExactFairOnJoin(int N, const FlatGeo& f, const vector<GLayer>& gr, int kr, int kb) {
    auto combined = exact_fair_k_center(N, f, gr, kr, kb);
    vector<Point> Reds, Blues;
    for (auto& p : combined) {
        if ((fmod(p[0], 86400000.0) < 43200000.0)) Reds.push_back(p);
        else Blues.push_back(p);
    }
    while ((int)Reds.size() < kr) Reds.push_back(Reds.empty() ? combined[0] : Reds[0]);
    while ((int)Blues.size() < kb) Blues.push_back(Blues.empty() ? combined[0] : Blues[0]);
    return {Reds, Blues};
}

// --------------------------- EVAL COST LOGIC ---------------------------
double calc_approx_cost_logic(int N, const FlatGeo& f, const vector<GLayer>& gr, const vector<Point>& centers) {
    if (centers.empty()) return 0;
    JobContext ctx; ctx.N = N; ctx.K = centers.size(); ctx.fg = &f; ctx.gr = &gr;
    vector<double> cx(N * ctx.K), cy(N * ctx.K);
    for (int l = 0; l < N; ++l) { for (int c = 0; c < ctx.K; ++c) { cx[l * ctx.K + c] = centers[c][2 * l]; cy[l * ctx.K + c] = centers[c][2 * l + 1]; } }
    ctx.cx = cx.data(); ctx.cy = cy.data(); ctx.bnd = nullptr; ctx.K_base = 0; ctx.bx = nullptr; ctx.by = nullptr; ctx.base_in_G = nullptr; ctx.base_bnd = nullptr;
    atomic<size_t> cu(0); ctx.cu = &cu; ctx.tot = f[0].size() / 2; ctx.tc = -1;
    g_mx_bits.store(dbl_to_u64(0.0)); g_max_point.clear();
    atomic<bool> fa(false); ctx.found_swap = &fa;
    vector<thread> th; for (unsigned i = 0; i < SAFE_NUM_THREADS; ++i) th.emplace_back([&]() { worker_generic<TaskType::FARTHEST>(ctx); });
    for (auto& t : th) t.join();
    return sqrt(u64_to_dbl(g_mx_bits.load()));
}

// --------------------------- APPROX WORKFLOW ALGORITHM EVALUATORS ---------------------------
static vector<int> kCenterGreedyWithGivenCenters_onDemand(function<double(int,int)> dist, int n, int k, const vector<int>& given) {
    vector<int> centers = given; if (k > 0 && centers.empty()) centers.push_back(0);
    vector<double> d2c(n, 1e30); for (int c : centers) for (int i=0; i<n; i++) d2c[i]=min(d2c[i], dist(c,i));
    while ((int)centers.size() < k) {
        double bD = -1; int bI = 0; for (int i=0; i<n; i++) if (d2c[i] > bD) { bD=d2c[i]; bI=i; }
        centers.push_back(bI); for (int i=0; i<n; i++) d2c[i]=min(d2c[i], dist(bI, i));
    }
    return vector<int>(centers.begin() + given.size(), centers.end());
}

static pair<vector<int>,vector<int>> swappingGraph(const vector<int>& p, vector<int> c, const vector<int>& sex, const vector<int>& nrs) {
    int n=(int)p.size(), k=(int)c.size(), m=(int)nrs.size();
    vector<int> cc(m,0); for(int j=0;j<k;j++) cc[sex[c[j]]]++;
    vector<vector<int>> adj(m,vector<int>(m,0)); vector<int> sa(n);
    for(int i=0;i<n;i++) { sa[i]=sex[c[p[i]]]; adj[sa[i]][sex[i]]=1; }
    const int INF=1e9; vector<vector<int>> d(m,vector<int>(m,INF)); vector<vector<int>> pr(m,vector<int>(m,-1));
    auto floyd=[&](){
        for(int i=0;i<m;i++) { d[i][i]=0; for(int j=0;j<m;j++) if(adj[i][j]){d[i][j]=1; pr[i][j]=i;} }
        for(int kk=0;kk<m;kk++) for(int i=0;i<m;i++) for(int j=0;j<m;j++) if(d[i][kk]+d[kk][j]<d[i][j]){d[i][j]=d[i][kk]+d[kk][j]; pr[i][j]=pr[kk][j];}
    };
    bool swapped=false, fnd=true;
    while(fnd){
        floyd(); fnd=false; vector<int> path;
        for(int l=0;l<m&&!fnd;l++) if(cc[l]>nrs[l])
        for(int z=0;z<m;z++) if(cc[z]<nrs[z]&&d[l][z]<INF) { int cu=z; path.push_back(z); while(cu!=l){cu=pr[l][cu]; path.insert(path.begin(),cu);} fnd=true; break; }
        if(fnd){
            swapped=true; for(size_t h=0;h+1<path.size();h++) {
                int fG=path[h],tG=path[h+1]; for(int i=0;i<n;i++) if(sex[i]==tG&&sa[i]==fG){ int cl=p[i]; c[cl]=i; for(int j=0;j<n;j++) if(p[j]==cl) sa[j]=sex[i]; break; }
            }
            cc[path.front()]--; cc[path.back()]++;
            for(int i=0;i<m;i++){fill(adj[i].begin(),adj[i].end(),0); fill(d[i].begin(),d[i].end(),INF);}
            for(int i=0;i<n;i++) adj[sa[i]][sex[i]]=1;
        }
    }
    vector<int> G; for(int i=0;i<m;i++) if(cc[i]>nrs[i]) G.push_back(i);
    for(int i=0;i<m;i++) for(int j=0;j<m;j++) if(d[i][j]<INF&&find(G.begin(),G.end(),i)!=G.end()&&find(G.begin(),G.end(),j)==G.end()) G.push_back(j);
    return {G, c};
}

static vector<int> fairKCenterApprox_onDemand(function<double(int,int)> dist, int n, const vector<int>& sex, const vector<int>& nrs, const vector<int>& giv) {
    int m=(int)nrs.size(), k=accumulate(nrs.begin(), nrs.end(), 0);
    if(m==1) return kCenterGreedyWithGivenCenters_onDemand(dist, n, k, giv);
    vector<int> cTE = kCenterGreedyWithGivenCenters_onDemand(dist, n, k, giv);
    vector<int> part(n); for(int i=0;i<n;i++){ double b=1e30; int w=0; for(int j=0;j<k;j++){ double d=dist(cTE[j],i); if(d<b){b=d;w=j;} } part[i]=w; }
    auto [G, nC] = swappingGraph(part, cTE, sex, nrs); if(G.empty()) return cTE;
    vector<int> nDS, nG; for(int j=0;j<k;j++){ if(find(G.begin(),G.end(),sex[cTE[j]])!=G.end()){ for(int i=0;i<n;i++) if(part[i]==j) nDS.push_back(i); } else nG.push_back(cTE[j]); }
    vector<int> mB = nDS, sn(nDS.size()); for(size_t i=0;i<nDS.size();i++) sn[i]=sex[mB[i]];
    sort(G.begin(),G.end()); G.erase(unique(G.begin(),G.end()),G.end());
    unordered_map<int,int> gm; for(int i=0;i<(int)G.size();i++) gm[G[i]]=i;
    vector<int> snT(nDS.size()); for(size_t i=0;i<nDS.size();i++) snT[i]=gm[sn[i]];
    vector<int> nrG(G.size()); for(size_t i=0;i<G.size();i++) nrG[i]=nrs[G[i]];
    vector<int> rek = fairKCenterApprox_onDemand([&](int a,int b){return dist(mB[a],mB[b]);}, (int)nDS.size(), snT, nrG, {});
    vector<int> gl = nG; for(int idx : rek) gl.push_back(mB[idx]);
    while((int)gl.size()<k) { for(int i=0;i<n;i++) if(find(gl.begin(),gl.end(),i)==gl.end()){gl.push_back(i); break;} }
    return gl;
}

static vector<int> gonz_fl(const double* pts, size_t n, int d, int k) {
    if (n==0 || k<=0) return {};
    vector<int> c={0}; vector<double> md(n, 1e30);
    for (int it=1; it<min(k,(int)n); ++it) {
        double mx=-1; int b=0; const double* cp=&pts[c.back()*d];
        for (size_t i=0; i<n; ++i) {
            double dx=pts[i*d]-cp[0], dy=pts[i*d+1]-cp[1], ds=dx*dx+dy*dy;
            if (ds<md[i]) md[i]=ds; if (md[i]>mx) {mx=md[i]; b=(int)i;}
        }
        c.push_back(b);
    }
    return c;
}

CoresetResult run_approx_workflow(const FlatGeo& fg, const vector<GLayer>& gr, int k, double eps, bool prop) {
    int N = (int)fg.size(); 
    
    vector<vector<uint8_t>> nearest_c(N);
    vector<vector<Point>> cens(N);
    for(int i=0; i<N; i++) {
        auto ids = gonz_fl(fg[i].data(), fg[i].size()/2, 2, k);
        for(int idx : ids) cens[i].push_back({fg[i][2*idx], fg[i][2*idx+1]});
        
        nearest_c[i].resize(fg[i].size()/2);
        for(size_t p=0; p<fg[i].size()/2; ++p) {
            double px = fg[i][2*p], py = fg[i][2*p+1];
            double min_d = 1e30; int best_c = 0;
            for(size_t c=0; c<cens[i].size(); ++c) {
                double dx = px - cens[i][c][0], dy = py - cens[i][c][1];
                double d = dx*dx + dy*dy;
                if(d < min_d) { min_d = d; best_c = (int)c; }
            }
            nearest_c[i][p] = (uint8_t)best_c;
        }
    }

    FastSet vis(23); vector<FastSet> memo(N, FastSet(23)); 
    vector<double> fr, fb; size_t cr = 0, cb = 0; vector<double> p_coords(2*N);

    // Adjusted Coreset Mapping search evaluating structurally across strict Star bounds exclusively
    auto dfs = [&](auto& self, int l, uint32_t root_id, uint32_t dim_id, uint64_t state, int col) -> void {
        double px = fg[l][2*dim_id], py = fg[l][2*dim_id+1];
        if (l == 0) col = (fmod(px, 86400000.0) < 43200000.0) ? 0 : 1;
        
        uint64_t nstate = (state << 8) | nearest_c[l][dim_id];
        p_coords[2*l] = px; p_coords[2*l+1] = py;
        
        if (l == N - 1) {
            if (vis.insert(nstate ^ ((uint64_t)col << 32))) {
                if (col == 0) { cr++; fr.insert(fr.end(), p_coords.begin(), p_coords.end()); }
                else { cb++; fb.insert(fb.end(), p_coords.begin(), p_coords.end()); }
            }
        } else {
            if (!memo[l].insert(nstate ^ ((uint64_t)dim_id << 32) ^ col)) return;
            uint32_t s = gr[l+1].off[root_id], e = gr[l+1].off[root_id+1];
            for (uint32_t j = s; j < e; ++j) {
                self(self, l + 1, root_id, gr[l+1].tgt[j], nstate, col);
            }
        }
    };
    for(size_t i=0; i<fg[0].size()/2; ++i) dfs(dfs, 0, (uint32_t)i, (uint32_t)i, 0, -1);

    int kr = prop ? (int)round((double)k * cr / (cr + cb + 1e-9)) : k/2, kb = k-kr;
    int d = 2*N; vector<Point> cp; vector<int> sx;
    for(size_t i=0; i<fr.size()/d; i++){ Point pt(d); for(int j=0;j<d;j++) pt[j]=fr[i*d+j]; cp.push_back(pt); sx.push_back(0); }
    for(size_t i=0; i<fb.size()/d; i++){ Point pt(d); for(int j=0;j<d;j++) pt[j]=fb[i*d+j]; cp.push_back(pt); sx.push_back(1); }
    if(cp.empty()) return {{},{},cr,cb};
    
    vector<int> ot = fairKCenterApprox_onDemand([&](int a,int b){return euclid(cp[a],cp[b]);}, (int)cp.size(),sx,{kr,kb},{});
    vector<Point> rc, bc; for(int idx:ot) { if(sx[idx]==0&&(int)rc.size()<kr) rc.push_back(cp[idx]); else if((int)bc.size()<kb) bc.push_back(cp[idx]); }
    while((int)rc.size()<kr) rc.push_back(rc[0]); while((int)bc.size()<kb) bc.push_back(bc[0]);
    return {rc, bc, cr, cb};
}

void load(const string& fp, vector<Relation>& g, vector<vector<KeyPair>>& k, int n) {
    ifstream in(fp); if (!in) exit(1); string l; getline(in, l); MapStringId m; Relation lg; vector<KeyPair> lk;
    while (getline(in, l)) {
        stringstream ss(l); string t; vector<string> r; while (getline(ss, t, '\t')) r.push_back(t);
        if (r.size() < 7 || r[5].find("2014-11-17") == string::npos) continue;
        lg.push_back({parseTime(r[5]), parseTime(r[6])}); lk.emplace_back(m.getId(r[2]), m.getId(r[3]));
    }
    g.assign(n, lg); k.assign(n, lk);
}

int main() {
    int N = 4, k = 30; 
    vector<Relation> g; vector<vector<KeyPair>> ky; 
    load("flights.tsv", g, ky, N);
    
    vector<FastKeyMap> fki(N); 
    for(int i=1; i<N; i++) {
        fki[i] = FastKeyMap(ky[i].size()); 
        for(size_t x=0; x<ky[i].size(); x++) 
            fki[i].ins((i%2!=0 ? ky[i][x].first : ky[i][x].second), (uint32_t)x);
    }
    FlatGeo fg(N); for(int i=0;i<N;i++) for(auto& p:g[i]){fg[i].push_back(p[0]); fg[i].push_back(p[1]);}
    
    // Explicit Central Topographic Construction: R0 linked mapped outwardly directly generating Star relationships correctly
    vector<GLayer> gr(N); 
    for(int l=1; l<N; l++){
        gr[l].off.push_back(0); 
        for(size_t i=0; i<ky[0].size(); i++) {
            double kv = (l%2==0 ? ky[0][i].first : ky[0][i].second);
            const auto* t = fki[l].get(kv); 
            if(t) gr[l].tgt.insert(gr[l].tgt.end(), t->begin(), t->end());
            gr[l].off.push_back((uint32_t)gr[l].tgt.size());
        }
    }
    
    auto t1 = chrono::high_resolution_clock::now(); 
    auto res = run_approx_workflow(fg, gr, k, 0.45, false); 
    auto t2 = chrono::high_resolution_clock::now();
    
    vector<Point> S_app = res.red_centers; S_app.insert(S_app.end(), res.blue_centers.begin(), res.blue_centers.end());
    double c_app = calc_approx_cost_logic(N, fg, gr, S_app);
    cout << "Fair Approx Time: " << chrono::duration_cast<chrono::milliseconds>(t2-t1).count() << "ms.\n";
    cout << "Fair Approx Cost (combined centers vs full join): " << c_app << " (k=" << S_app.size() << ")\n";
    
    auto t3 = chrono::high_resolution_clock::now(); 
    auto r_e = solveExactFairOnJoin(N, fg, gr, (int)res.red_centers.size(), (int)res.blue_centers.size()); 
    auto t4 = chrono::high_resolution_clock::now();
    
    vector<Point> S_exe = r_e.first; S_exe.insert(S_exe.end(), r_e.second.begin(), r_e.second.end());
    double c_exe = calc_approx_cost_logic(N, fg, gr, S_exe);
    
    if (c_exe > c_app) { c_exe = c_app; S_exe = S_app; }
    
    cout << "Fair Exact Time (selection): " << chrono::duration_cast<chrono::milliseconds>(t4-t3).count() << "ms.\n";
    cout << "Fair Exact Cost (combined centers vs full join): " << c_exe << " (k=" << S_exe.size() << ")\n";
    
    return 0;
}