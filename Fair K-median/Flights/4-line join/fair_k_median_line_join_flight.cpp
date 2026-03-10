// fair_k_median_line_join_hyperfast.cpp
//
// OPTIMIZATIONS:
// 1. Delta-Cost Solver Updates (O(N) vs O(NK)).
// 2. Spatial Deduplication (Smaller Coreset).
// 3. Flattened Memory Access.
// 4. Combined color counting in countRectBoth (halves DP runs).
// 5. Minor micro-optimizations (emplace_back, reserve).
//
// Compile:
// g++ -O3 -pthread -march=native -funroll-loops --std=c++20 fair_k_median_line_join_hyperfast.cpp -o aa.out

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <chrono>
#include <limits>
#include <numeric>

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
// 2. LOADING
// -------------------------
struct Row { int src, dst; double t1, t2; };
double pTime(string s) {
    if (s.size() < 19) return 0;
    int dd = stoi(s.substr(8, 2)), h = stoi(s.substr(11, 2)), m = stoi(s.substr(14, 2));
    return (dd * 86400LL + h * 3600 + m * 60) * 1000.0;
}
class IDMapper {
    unordered_map<string, int> m; int c = 0;
public:
    int get(string s) { if (!m.count(s)) m[s] = c++; return m[s]; }
    int count() const { return c; }
};
int MAX_KEY_ID = 0;
void Load(string f, vector<Row>& out) {
    ifstream in(f); if (!in) exit(1);
    string l; getline(in, l); IDMapper m_gen;
    while (getline(in, l)) {
        stringstream ss(l); string v; vector<string> c; while (getline(ss, v, '\t')) c.push_back(v);
        if (c.size() > 6 && c[5].find("2014-11-17") != string::npos) 
            out.push_back({ m_gen.get(c[2]), m_gen.get(c[3]), pTime(c[5]), pTime(c[6]) });
    }
    MAX_KEY_ID = m_gen.count();
    cout << "Loaded Rows: " << out.size() << " | Max Key ID: " << MAX_KEY_ID << endl;
}

// -------------------------
// 3. RELATIONAL ENGINE (Optimized)
// -------------------------
struct RelTuple { int jl, jr; double t1, t2; int c; };
static bool comp_t1(const RelTuple& a, double val) { return a.t1 < val; }

class LineEngine {
    vector<RelTuple> tables[4];
    RNG rng;
public:
    void build(const vector<Row>& d) {
        tables[0].reserve(d.size()); tables[1].reserve(d.size());
        tables[2].reserve(d.size()); tables[3].reserve(d.size());
        for (auto& r : d) {
            int col = (fmod(r.t1, 86400000.0) < 43200000.0) ? 0 : 1;
            tables[0].emplace_back(-1, r.dst, r.t1, r.t2, col);
            tables[1].emplace_back(r.src, r.dst, r.t1, r.t2, col);
            tables[2].emplace_back(r.src, r.dst, r.t1, r.t2, col);
            tables[3].emplace_back(r.src, -1, r.t1, r.t2, col);
        }
        for (int i = 0; i < 4; ++i) sort(tables[i].begin(), tables[i].end(), [](const RelTuple& a, const RelTuple& b) { return a.t1 < b.t1; });
    }

    long long countRect(const Box& b, int color, int r_start, int r_end) {
        int num_rels = r_end - r_start;
        static vector<vector<long long>> dp;
        if(dp.size() < num_rels) dp.resize(num_rels);
        for(auto& v : dp) { if(v.size() < MAX_KEY_ID) v.resize(MAX_KEY_ID, 0); else fill(v.begin(), v.end(), 0); }

        int r_curr = r_end - 1;
        int dim_off = (r_curr - r_start) * 2;
        auto it_l = lower_bound(tables[r_curr].begin(), tables[r_curr].end(), b.bounds[dim_off].first, comp_t1);
        auto it_r = lower_bound(tables[r_curr].begin(), tables[r_curr].end(), b.bounds[dim_off].second, comp_t1);

        for (auto it = it_l; it != it_r; ++it) {
            if (it->t2 >= b.bounds[dim_off+1].first && it->t2 < b.bounds[dim_off+1].second) {
                if (r_curr == 3 && color != -1 && it->c != color) continue;
                if (it->jl >= 0) dp[r_curr - r_start][it->jl]++;
            }
        }
        for (int r = r_end - 2; r >= r_start; --r) {
            int d_off = (r - r_start) * 2;
            int dp_idx = r - r_start;
            auto it_left = lower_bound(tables[r].begin(), tables[r].end(), b.bounds[d_off].first, comp_t1);
            auto it_right = lower_bound(tables[r].begin(), tables[r].end(), b.bounds[d_off].second, comp_t1);
            for (auto it = it_left; it != it_right; ++it) {
                if (it->t2 >= b.bounds[d_off+1].first && it->t2 < b.bounds[d_off+1].second) {
                    if (it->jr >= 0 && dp[dp_idx + 1][it->jr] > 0) {
                        int jl = (it->jl == -1) ? 0 : it->jl;
                        if(jl >= 0) dp[dp_idx][jl] += dp[dp_idx + 1][it->jr];
                    }
                }
            }
        }
        if (r_start == 0) return dp[0][0];
        long long tot = 0; for (long long v : dp[0]) tot += v;
        return tot;
    }

    // Count both colors in one DP pass
    pair<long long, long long> countRectBoth(const Box& b, int r_start, int r_end) {
        int num_rels = r_end - r_start;
        static vector<vector<pair<long long, long long>>> dp;
        if (dp.size() < num_rels) dp.resize(num_rels);
        for (auto& v : dp) {
            if (v.size() < MAX_KEY_ID) v.resize(MAX_KEY_ID, {0,0});
            else fill(v.begin(), v.end(), make_pair(0LL,0LL));
        }

        int r_curr = r_end - 1;
        int dim_off = (r_curr - r_start) * 2;
        auto it_l = lower_bound(tables[r_curr].begin(), tables[r_curr].end(), b.bounds[dim_off].first, comp_t1);
        auto it_r = lower_bound(tables[r_curr].begin(), tables[r_curr].end(), b.bounds[dim_off].second, comp_t1);

        for (auto it = it_l; it != it_r; ++it) {
            if (it->t2 >= b.bounds[dim_off+1].first && it->t2 < b.bounds[dim_off+1].second) {
                if (it->jl >= 0) {
                    auto& entry = dp[r_curr - r_start][it->jl];
                    if (it->c == 0) entry.first++;
                    else entry.second++;
                }
            }
        }
        for (int r = r_end - 2; r >= r_start; --r) {
            int d_off = (r - r_start) * 2;
            int dp_idx = r - r_start;
            auto it_left = lower_bound(tables[r].begin(), tables[r].end(), b.bounds[d_off].first, comp_t1);
            auto it_right = lower_bound(tables[r].begin(), tables[r].end(), b.bounds[d_off].second, comp_t1);
            for (auto it = it_left; it != it_right; ++it) {
                if (it->t2 >= b.bounds[d_off+1].first && it->t2 < b.bounds[d_off+1].second) {
                    if (it->jr >= 0) {
                        auto add = dp[dp_idx + 1][it->jr];
                        if (add.first > 0 || add.second > 0) {
                            int jl = (it->jl == -1) ? 0 : it->jl;
                            dp[dp_idx][jl].first += add.first;
                            dp[dp_idx][jl].second += add.second;
                        }
                    }
                }
            }
        }
        if (r_start == 0) return dp[0][0];
        long long tot0 = 0, tot1 = 0;
        for (auto& p : dp[0]) { tot0 += p.first; tot1 += p.second; }
        return {tot0, tot1};
    }

    // Original sampleBatch (fast, used for both colors separately)
    void sampleBatch(const Box& b, int M_req, int color, int r_start, int r_end, vector<Point>& out) {
        out.clear();
        if (M_req == 0) return;
        out.reserve(M_req);

        int num_rels = r_end - r_start;
        static vector<vector<long long>> dp;
        static vector<vector<const RelTuple*>> valid;
        static vector<vector<vector<const RelTuple*>>> grouped;

        dp.resize(num_rels);
        for (int i = 0; i < num_rels; ++i) {
            if (dp[i].size() < MAX_KEY_ID) dp[i].resize(MAX_KEY_ID);
            fill(dp[i].begin(), dp[i].end(), 0);
        }

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

        for (auto it = it_l; it != it_r; ++it) {
            if (it->t2 >= b.bounds[dim_off+1].first && it->t2 < b.bounds[dim_off+1].second) {
                if (r_curr == 3 && color != -1 && it->c != color) continue;
                if (it->jl >= 0) { dp[r_curr - r_start][it->jl]++; valid[r_curr - r_start].push_back(&*it); }
            }
        }
        for (int r = r_end - 2; r >= r_start; --r) {
            int d_off = (r - r_start) * 2;
            int dp_idx = r - r_start;
            auto it_left = lower_bound(tables[r].begin(), tables[r].end(), b.bounds[d_off].first, comp_t1);
            auto it_right = lower_bound(tables[r].begin(), tables[r].end(), b.bounds[d_off].second, comp_t1);
            for (auto it = it_left; it != it_right; ++it) {
                if (it->t2 >= b.bounds[d_off+1].first && it->t2 < b.bounds[d_off+1].second) {
                    if (it->jr >= 0 && dp[dp_idx + 1][it->jr] > 0) {
                        int jl = (it->jl == -1) ? 0 : it->jl;
                        dp[dp_idx][jl] += dp[dp_idx + 1][it->jr];
                        valid[dp_idx].push_back(&*it);
                    }
                }
            }
        }
        long long total_w = 0;
        if (r_start == 0) total_w = dp[0][0];
        else for (long long v : dp[0]) total_w += v;
        if (total_w == 0) return;

        for (int i = 0; i < num_rels; ++i) {
            for (auto ptr : valid[i]) {
                int key = (ptr->jl == -1) ? 0 : ptr->jl;
                grouped[i][key].push_back(ptr);
            }
        }

        vector<double> random_picks(M_req);
        for(int m=0; m<M_req; ++m) random_picks[m] = rng.unit() * total_w;
        sort(random_picks.begin(), random_picks.end());

        int pick_idx = 0;
        const vector<const RelTuple*>* root_cands = &valid[0];
        vector<int> current_links(M_req);
        long long run_sum = 0;
        for (auto ptr : *root_cands) {
            long long w = 1;
            if (num_rels > 1) w = dp[1][ptr->jr];
            while (pick_idx < M_req && random_picks[pick_idx] < run_sum + w) {
                Point pt; pt.reserve(num_rels*2);
                pt.push_back(ptr->t1); pt.push_back(ptr->t2);
                out.push_back(std::move(pt));
                current_links[pick_idx] = ptr->jr;
                pick_idx++;
            }
            run_sum += w;
            if (pick_idx >= M_req) break;
        }

        for (int r = r_start + 1; r < r_end; ++r) {
            int idx = r - r_start;
            for(int m=0; m<out.size(); ++m) {
                int link = current_links[m];
                const vector<const RelTuple*>& cands = grouped[idx][link];
                const RelTuple* selected = cands[0];
                if(cands.size() > 1) selected = cands[rng.idx(cands.size())];
                out[m].push_back(selected->t1); out[m].push_back(selected->t2);
                current_links[m] = selected->jr;
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
                for(const auto& old : B) if(old.Intersects(bx)) relevant_B.push_back(&old);
                Box brw = bx; 
                for (auto& iv : brw.bounds) { 
                    iv.first *= 86400000.0; 
                    iv.second *= 86400000.0; 
                }
                
                bool added_heavy = false;
                // Use combined count for both colors
                auto [n_box0, n_box1] = db.countRectBoth(brw, r_start, r_end);
                for (int color : {0, 1}) {
                    long long n_box = (color == 0) ? n_box0 : n_box1;
                    if (n_box > 0) {
                        vector<Point> S_batch; 
                        db.sampleBatch(brw, M, color, r_start, r_end, S_batch);
                        int fresh = 0; Point rep;
                        for (auto& s_raw : S_batch) {
                            Point s = s_raw; Scale(s);
                            bool covered = false; 
                            for (const auto* old : relevant_B) if (old->Contains(s)) { covered = true; break; }
                            if (!covered) { fresh++; if (rep.empty()) rep = s; }
                        }
                        if ((double)fresh / M >= 2.0 * tau || (relevant_B.empty() && fresh > 0)) {
                            bool merged = false;
                            for(auto& ex : C) {
                                if(ex.color == color && L1Dist(ex.point, rep) < side * 0.1) {
                                    ex.weight += (double)n_box * ((double)fresh / M) * (1.0 / (1.0 - eps_u));
                                    merged = true; break;
                                }
                            }
                            if(!merged) C.emplace_back(rep, (double)n_box * ((double)fresh / M) * (1.0 / (1.0 - eps_u)), color);
                            added_heavy = true;
                        }
                    }
                }
                if(added_heavy) B.push_back(bx);
            }
        }
    }
    double tot_w = 0; for(auto& wp : C) tot_w += wp.weight;
    if(tot_w > 0) { double sc = (double)N / tot_w; for(auto& wp : C) wp.weight *= sc; }
    return C;
}

// -------------------------
// 5. SOLVERS (Hyper-Optimized)
// -------------------------
double WeightedCost(const vector<WeightedPoint>& C, const vector<Point>& S) {
    double t = 0; for (auto& p : C) { double m = 1e18; for (auto& s : S) m = min(m, L1Dist(p.point, s)); t += p.weight * m; } return t;
}

void KMeansPlusPlusInit(const vector<WeightedPoint>& C, int k, vector<Point>& res, RNG& r) {
    if (C.empty() || k <= 0) return;
    res.clear(); res.reserve(k);
    double total_w = 0; for(auto& p : C) total_w += p.weight;
    double pick = r.unit() * total_w; double s = 0;
    for(auto& p : C) { s += p.weight; if(s >= pick) { res.push_back(p.point); break; } }
    if(res.empty()) res.push_back(C.back().point);
    for(int i=1; i<k; ++i) {
        vector<double> dists; double sum_d = 0;
        dists.reserve(C.size());
        for(auto& p : C) {
            double d = 1e18; for(auto& c : res) d = min(d, L1Dist(p.point, c));
            double prob_weight = p.weight * d; sum_d += prob_weight; dists.push_back(prob_weight);
        }
        if(sum_d == 0) { res.push_back(C[r.idx(C.size())].point); continue; }
        double p_val = r.unit() * sum_d; s = 0;
        for(size_t j=0; j<dists.size(); ++j) { s += dists[j]; if(s >= p_val) { res.push_back(C[j].point); break; } }
    }
}

void solveStandard(const vector<WeightedPoint>& C, int k, vector<Point>& result) {
    result.clear(); if(C.empty() || k==0) return;
    RNG r(777); 
    vector<Point> best_res; double best_cost = 1e18;
    for(int attempt=0; attempt<3; ++attempt) {
        vector<Point> current; KMeansPlusPlusInit(C, k, current, r);
        vector<double> min_dists(C.size(), 1e18);
        vector<int> assignments(C.size(), -1);
        for(size_t i=0; i<C.size(); ++i) {
            for(int j=0; j<k; ++j) {
                double d = L1Dist(C[i].point, current[j]);
                if(d < min_dists[i]) { min_dists[i] = d; assignments[i] = j; }
            }
        }
        double cur_cost = 0; for(size_t i=0; i<C.size(); ++i) cur_cost += C[i].weight * min_dists[i];
        
        for (int iter = 0; iter < 40; ++iter) {
            bool mod = false;
            for (int j = 0; j < current.size(); ++j) {
                int id = r.idx(C.size()); Point candidate = C[id].point;
                double delta = 0;
                for(size_t p=0; p<C.size(); ++p) {
                    double d_new = L1Dist(C[p].point, candidate);
                    double d_old = min_dists[p];
                    if (assignments[p] == j) {
                        double second_best = 1e18;
                        for(int c=0; c<k; ++c) if(c != j) second_best = min(second_best, L1Dist(C[p].point, current[c]));
                        delta += C[p].weight * (min(d_new, second_best) - d_old);
                    } else {
                        if (d_new < d_old) delta += C[p].weight * (d_new - d_old);
                    }
                }
                if (delta < 0) {
                    current[j] = candidate; cur_cost += delta; mod = true;
                    for(size_t p=0; p<C.size(); ++p) {
                        min_dists[p] = 1e18;
                        for(int c=0; c<k; ++c) {
                            double d = L1Dist(C[p].point, current[c]);
                            if(d < min_dists[p]) { min_dists[p] = d; assignments[p] = c; }
                        }
                    }
                }
            }
            if (!mod) break;
        }
        if(cur_cost < best_cost) { best_cost = cur_cost; best_res = current; }
    }
    result = best_res;
}

void solveFair(const vector<WeightedPoint>& C, int kr, int kb, vector<Point>& result) {
    vector<WeightedPoint> CR, CB; 
    CR.reserve(C.size()); CB.reserve(C.size());
    for(auto& p : C) if(p.color==0) CR.push_back(p); else CB.push_back(p);
    RNG r(777); 
    vector<Point> best_res; double best_cost = 1e18;
    for(int attempt=0; attempt<3; ++attempt) {
        vector<Point> solR, solB;
        KMeansPlusPlusInit(CR, kr, solR, r); KMeansPlusPlusInit(CB, kb, solB, r);
        vector<Point> current = solR; current.insert(current.end(), solB.begin(), solB.end());
        double cur_cost = WeightedCost(C, current);
        if(cur_cost < best_cost) { best_cost = cur_cost; best_res = current; }
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
    vector<Row> raw; Load("flights.tsv", raw); if (raw.empty()) return 1;
    LineEngine db; db.build(raw);

    for(int k_goal : k_values) {
        cout << "\n-----------------------------------" << endl;
        cout << "Running Algorithm 3 (Line Join) for k = " << k_goal << endl;
        auto t1 = chrono::high_resolution_clock::now();

        auto C12 = Algorithm2(db, k_goal, eps, 0, 2, {});
        vector<Point> S12; solveStandard(C12, k_goal, S12); 

        auto C34 = Algorithm2(db, k_goal, eps, 2, 4, {});
        vector<Point> S34; solveStandard(C34, k_goal, S34); 

        vector<Point> X_root;
        X_root.reserve(S12.size() * S34.size());
        for(auto& l : S12) for(auto& r_p : S34) {
            Point p = l; p.insert(p.end(), r_p.begin(), r_p.end()); X_root.push_back(p);
        }
        auto Coreset = Algorithm2(db, k_goal, eps, 0, 4, X_root);

        int kr = k_goal / 2, kb = k_goal - kr;
        if (proportional && !Coreset.empty()) {
            Box inf; for (int i = 0; i < 8; ++i) inf.bounds.emplace_back(-1e18, 1e18);
            long long nR = db.countRect(inf, 0, 0, 4), nB = db.countRect(inf, 1, 0, 4);
            if(nR+nB>0) { 
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