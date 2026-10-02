// Fixed-operator fourth-order symmetric Hermite / [2/2] Pade evolution,
// or third-order L-stable two-stage Radau IIA ([1/2] Pade).
// u_t = v, v_t = G v - J u.  All input decimals and arithmetic use QD
// double-double unless USE_DOUBLE is explicitly selected at build time.
// The two conjugate Cayley factors are applied in increment form to avoid
// subtracting nearly equal full states in small-amplitude tails.
#ifndef USE_DOUBLE
#include <qd/dd_real.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
#ifdef USE_DOUBLE
using R = double;
constexpr const char* precision_name = "binary64";
bool finite(R x) { return std::isfinite(x); }
double as_double(R x) { return x; }
#else
using R = dd_real;
constexpr const char* precision_name = "double-double";
bool finite(const R& x) { return x.isfinite(); }
double as_double(const R& x) { return to_double(x); }
#endif
size_t decimal_underflow_count = 0;
size_t decimal_subnormal_count = 0;
R decimal(const std::string& s) {
    // QD adds precision, not exponent range. strtod is used only to classify
    // exponent-range failures; normal values are then parsed independently
    // at full precision. QD's raw reader can overflow an intermediate power
    // even for representable decimals such as a 50-digit mantissa times 1e-300.
    char* end = nullptr;
    double range_check = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0' || !std::isfinite(range_check))
        throw std::runtime_error("Invalid or overflowing decimal: " + s);
    size_t position = 0;
    bool negative = false;
    if (position < s.size() && (s[position] == '+' || s[position] == '-')) {
        negative = s[position] == '-'; ++position;
    }
    std::string digits;
    long long before_point = 0;
    bool has_point = false;
    while (position < s.size() && s[position] != 'e' && s[position] != 'E') {
        char ch = s[position++];
        if (ch == '.' && !has_point) { has_point = true; continue; }
        if (ch < '0' || ch > '9') throw std::runtime_error("Invalid decimal: " + s);
        digits += ch;
        if (!has_point) ++before_point;
    }
    if (digits.empty()) throw std::runtime_error("Invalid decimal: " + s);
    size_t first = digits.find_first_not_of('0');
    if (first == std::string::npos) return R(0);
    if (range_check == 0.0) { ++decimal_underflow_count; return R(0); }
    // Below the normal binary64 range a second component cannot extend the
    // smallest representable spacing. Correctly rounded strtod is sufficient.
    if (std::abs(range_check) < std::numeric_limits<double>::min()) {
        ++decimal_subnormal_count; return R(range_check);
    }
#ifdef USE_DOUBLE
    return range_check;
#else
    long long exponent = position == s.size() ? 0 : std::stoll(s.substr(position+1));
    exponent += before_point - static_cast<long long>(first) - 1;
    std::string normalized = (negative ? "-" : "") + digits.substr(first,1)
                           + "." + digits.substr(first+1);
    R value;
    if (value.read(normalized.c_str(),value) != 0)
        throw std::runtime_error("Invalid normalized decimal: " + s);
    while (exponent != 0) {
        int chunk = static_cast<int>(std::max(-100LL,std::min(100LL,exponent)));
        value *= (R(10)^chunk);
        exponent -= chunk;
    }
    if (!finite(value)) throw std::runtime_error("Nonfinite parsed decimal: " + s);
    return value;
#endif
}
using std::abs;
using std::sqrt;
using std::floor;
using Clock = std::chrono::steady_clock;
enum class Method { hermite, radau3 };

struct C {
    R re = R(0), im = R(0);
    C() = default;
    C(const R& x) : re(x) {}
    C(const R& x, const R& y) : re(x), im(y) {}
    C& operator+=(const C& b) { re += b.re; im += b.im; return *this; }
    C& operator-=(const C& b) { re -= b.re; im -= b.im; return *this; }
};
C operator+(C a, const C& b) { return a += b; }
C operator-(C a, const C& b) { return a -= b; }
C operator*(const C& a, const C& b) {
    return C(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re);
}
C operator*(const R& a, const C& b) { return C(a*b.re, a*b.im); }
C operator*(const C& a, const R& b) { return b*a; }
C operator/(const C& a, const C& b) {
    // Smith division avoids squaring the denominator's magnitude.
    if (abs(b.re) >= abs(b.im)) {
        R r = b.im/b.re, den = b.re + b.im*r;
        return C((a.re + a.im*r)/den, (a.im - a.re*r)/den);
    }
    R r = b.re/b.im, den = b.im + b.re*r;
    return C((a.re*r + a.im)/den, (a.im*r - a.re)/den);
}
C conjugate(const C& a) { return C(a.re, -a.im); }
bool zero(const C& a) { return a.re == R(0) && a.im == R(0); }
R magnitude_bound(const C& a) { return abs(a.re) + abs(a.im); }
using V = std::vector<C>;
C pole_coefficient(const R& dt, Method method) {
    if (method == Method::radau3)
        return C(dt/R(3), dt/(R(3)*sqrt(R(2))));
    return C(dt)/C(R(3),sqrt(R(3)));
}

R read_real(std::istream& stream) {
    std::string token;
    if (!(stream >> token)) throw std::runtime_error("Truncated input");
    R x = decimal(token);
    if (!finite(x)) throw std::runtime_error("Nonfinite input");
    return x;
}

struct Problem {
    int n, bw;
    std::vector<R> rho, j, g;
    V u, v;
    explicit Problem(const std::filesystem::path& file) {
        std::ifstream in(file);
        if (!(in >> n >> bw) || n < 1 || n > 100000 || bw < 0 || bw >= n ||
            static_cast<long long>(n)*(2*bw + 1) > 100000000)
            throw std::runtime_error("Invalid matrix size or bandwidth");
        rho.resize(n); u.resize(n); v.resize(n);
        j.assign(static_cast<size_t>(n)*(2*bw + 1), R(0));
        g = j;
        for (int i = 0; i < n; ++i) {
            rho[i] = read_real(in); u[i] = C(read_real(in)); v[i] = C(read_real(in));
        }
        for (int i = 0; i < n; ++i)
            for (int k = std::max(0, i-bw); k < std::min(n, i+bw+1); ++k) {
                j[offset(i,k)] = read_real(in); g[offset(i,k)] = read_real(in);
            }
        std::string extra;
        if (in >> extra) throw std::runtime_error("Extra input after band matrices");
    }
    size_t offset(int i, int k) const { return static_cast<size_t>(i)*(2*bw+1) + k-i+bw; }
    // Evaluate J u and G v in a single traversal of the band.
    void products(const V& a, const V& b, V& ja, V& gb) const {
        ja.assign(n, C()); gb.assign(n, C());
        for (int i = 0; i < n; ++i)
            for (int k = std::max(0, i-bw); k < std::min(n, i+bw+1); ++k) {
                size_t p = offset(i,k);
                ja[i] += j[p]*a[k]; gb[i] += g[p]*b[k];
            }
    }
    V j_product(const V& a) const {
        V out(n);
        for (int i = 0; i < n; ++i)
            for (int k = std::max(0, i-bw); k < std::min(n, i+bw+1); ++k)
                out[i] += j[offset(i,k)]*a[k];
        return out;
    }
};

// General band LU with partial row pivoting. Pivoting extends the upper
// bandwidth to 2*bw. Multipliers are stored in elimination order; the same
// swaps are applied to each right-hand side before each elimination column.
struct Factor {
    const int n, bw, stride;
    const C alpha;
    V mat, lower;
    std::vector<int> pivot, last_lower, last_upper;
    C& at(int i, int j) {
        if (i < 0 || i >= n || j < 0 || j >= n || j-i < -bw || j-i > 2*bw)
            throw std::runtime_error("Internal band-factor index error");
        return mat[static_cast<size_t>(i)*stride + j-i+bw];
    }
    const C& at(int i, int j) const { return mat[static_cast<size_t>(i)*stride + j-i+bw]; }
    Factor(const Problem& p, const R& dt, Method method = Method::hermite)
        : n(p.n), bw(p.bw), stride(3*bw+1),
          alpha(pole_coefficient(dt,method)),
          mat(static_cast<size_t>(n)*stride), lower(static_cast<size_t>(n)*bw),
          pivot(n), last_lower(n), last_upper(n) {
        C a2 = alpha*alpha;
        for (int i = 0; i < n; ++i) {
            for (int j = std::max(0,i-bw); j < std::min(n,i+bw+1); ++j)
                at(i,j) = a2*p.j[p.offset(i,j)] - alpha*p.g[p.offset(i,j)];
            at(i,i) += C(R(1));
        }
        for (int k = 0; k < n; ++k) {
            int row = k;
            R best = magnitude_bound(at(k,k));
            for (int i = k+1; i < std::min(n,k+bw+1); ++i) {
                R value = magnitude_bound(at(i,k));
                if (value > best) { row = i; best = value; }
            }
            if (!finite(best) || best == R(0)) throw std::runtime_error("Singular or nonfinite LU pivot");
            pivot[k] = row;
            int top = std::min(n,k+2*bw+1);
            if (row != k)
                for (int j = k; j < top; ++j) std::swap(at(k,j), at(row,j));
            last_lower[k] = k;
            for (int i = k+1; i < std::min(n,k+bw+1); ++i) {
                C mult = at(i,k)/at(k,k);
                lower[static_cast<size_t>(k)*bw + i-k-1] = mult;
                at(i,k) = C();
                if (zero(mult)) continue;
                last_lower[k] = i;
                for (int j = k+1; j < top; ++j) at(i,j) -= mult*at(k,j);
            }
            last_upper[k] = k;
            for (int j = k+1; j < top; ++j)
                if (!zero(at(k,j))) last_upper[k] = j;
        }
    }
    V solve(V rhs) const {
        for (int k = 0; k < n; ++k) {
            std::swap(rhs[k], rhs[pivot[k]]);
            for (int i = k+1; i <= last_lower[k]; ++i)
                rhs[i] -= lower[static_cast<size_t>(k)*bw + i-k-1]*rhs[k];
        }
        for (int k = n-1; k >= 0; --k) {
            for (int j = k+1; j <= last_upper[k]; ++j) rhs[k] -= at(k,j)*rhs[j];
            rhs[k] = rhs[k]/at(k,k);
        }
        return rhs;
    }
};

void substep(const Problem& p, const Factor& f, V& u, V& v, bool conjugated) {
    C alpha = conjugated ? conjugate(f.alpha) : f.alpha;
    V ju, gv;
    p.products(u,v,ju,gv);
    V ru(p.n), rv(p.n);
    for (int i = 0; i < p.n; ++i) {
        ru[i] = R(2)*alpha*v[i];
        rv[i] = R(2)*alpha*(gv[i]-ju[i]);
    }
    V jru = p.j_product(ru);
    for (int i = 0; i < p.n; ++i) {
        rv[i] -= alpha*jru[i];
        if (conjugated) rv[i] = conjugate(rv[i]);
    }
    V dv = f.solve(std::move(rv));
    for (int i = 0; i < p.n; ++i) {
        if (conjugated) dv[i] = conjugate(dv[i]);
        u[i] += ru[i] + alpha*dv[i];
        v[i] += dv[i];
    }
}

// Apply (I-alpha A)^(-1) to (u,v), using the same Schur factorization as
// Hermite. This is a full-state solve, not the Cayley increment. No real
// projection occurs until both conjugate resolvents have been applied.
void resolvent(const Problem& p, const Factor& f, V& u, V& v, bool conjugated) {
    C alpha = conjugated ? conjugate(f.alpha) : f.alpha;
    V ju = p.j_product(u), rhs(p.n);
    for (int i = 0; i < p.n; ++i) {
        rhs[i] = v[i]-alpha*ju[i];
        if (conjugated) rhs[i] = conjugate(rhs[i]);
    }
    V new_v = f.solve(std::move(rhs));
    for (int i = 0; i < p.n; ++i) {
        if (conjugated) new_v[i] = conjugate(new_v[i]);
        u[i] += alpha*new_v[i];
        v[i] = new_v[i];
    }
}

void radau_step(const Problem& p, const Factor& f, const R& dt, V& u, V& v) {
    // R(z)=(1+z/3)/(1-2z/3+z^2/6). All factors commute for the fixed
    // linear autonomous operator. The numerator must use the old state.
    V ju, gv;
    p.products(u,v,ju,gv);
    R third_dt = dt/R(3);
    for (int i = 0; i < p.n; ++i) {
        u[i] += third_dt*v[i];
        v[i] += third_dt*(gv[i]-ju[i]);
    }
    resolvent(p,f,u,v,false); resolvent(p,f,u,v,true);
}

std::ofstream output(const std::filesystem::path& directory, const char* name) {
    std::ofstream stream(directory/name);
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream << std::scientific << std::setprecision(32);
    return stream;
}
} // namespace

int main(int argc, char** argv) try {
    if (argc < 6)
        throw std::runtime_error("Usage: hermite_banded input output_directory dt endtime output_every_steps [max_runtime_seconds] [probe_indices...] [--integrator hermite|radau3]");
    R dt = decimal(argv[3]), end = decimal(argv[4]);
    int cadence = std::stoi(argv[5]);
    Method method = Method::hermite;
    bool method_given = false;
    std::vector<std::string> optional;
    for (int k = 6; k < argc; ++k) {
        std::string argument(argv[k]);
        if (argument == "--integrator") {
            if (method_given || ++k >= argc)
                throw std::runtime_error("Give --integrator once, followed by hermite or radau3");
            method_given = true;
            std::string choice(argv[k]);
            if (choice == "radau3") method = Method::radau3;
            else if (choice != "hermite") throw std::runtime_error("Unknown integrator: " + choice);
        } else if (argument.rfind("--",0) == 0) {
            throw std::runtime_error("Unknown option: " + argument);
        } else optional.push_back(argument);
    }
    double runtime_limit = optional.empty() ? 3600.0 : std::stod(optional.front());
    if (!finite(dt) || !finite(end) || dt <= R(0) || end < R(0) || cadence <= 0 ||
        !std::isfinite(runtime_limit) || runtime_limit <= 0 || end/dt > R(2000000000))
        throw std::runtime_error("Invalid timestep, end time, cadence, or runtime limit");
    Problem p(argv[1]);
    std::vector<int> probes;
    for (size_t k = 1; k < optional.size(); ++k) {
        int index = std::stoi(optional[k]);
        if (index < 0 || index >= p.n) throw std::runtime_error("Probe index outside grid");
        probes.push_back(index);
    }
    const std::filesystem::path directory(argv[2]);
    if (std::filesystem::exists(directory)) throw std::runtime_error("Output destination already exists");
    if (!std::filesystem::create_directory(directory)) throw std::runtime_error("Cannot create output destination");
    auto history = output(directory,"history.csv");
    history << "tau,u_left,v_left,u_right,v_right";
    for (int index : probes) history << ",u_" << index << ",v_" << index;
    history << '\n';
    int steps = static_cast<int>(as_double(floor(end/dt)));
    R remainder = end - R(steps)*dt;
#ifdef USE_DOUBLE
    R tolerance = R(64)*std::numeric_limits<double>::epsilon()*std::max(end,dt);
#else
    R tolerance = decimal("1e-29")*std::max(end,dt);
#endif
    if (abs(remainder-dt) < tolerance) { ++steps; remainder = R(0); }
    if (abs(remainder) < tolerance) remainder = R(0);
    const auto start = Clock::now();
    std::unique_ptr<Factor> factor;
    if (steps > 0) factor = std::make_unique<Factor>(p,dt,method);
    const auto setup_done = Clock::now();
    V u = p.u, v = p.v;
    R imaginary_max = R(0), state_max = R(0);
    auto record = [&](const R& tau) {
        history << tau << ',' << u.front().re << ',' << v.front().re << ','
                << u.back().re << ',' << v.back().re;
        for (int index : probes) history << ',' << u[index].re << ',' << v[index].re;
        history << '\n';
    };
    auto advance = [&](const Factor& f, const R& step_dt) {
        if (method == Method::radau3) radau_step(p,f,step_dt,u,v);
        else { substep(p,f,u,v,false); substep(p,f,u,v,true); }
        for (V* row : {&u,&v}) for (C& value : *row) {
            if (!finite(value.re) || !finite(value.im)) throw std::runtime_error("Nonfinite solution");
            imaginary_max = std::max(imaginary_max, R(abs(value.im)));
            state_max = std::max(state_max, R(abs(value.re)));
            value.im = R(0);
        }
    };
    record(R(0));
    for (int k = 1; k <= steps; ++k) {
        advance(*factor,dt);
        if (k%cadence == 0 || (k == steps && remainder == R(0))) record(R(k)*dt);
        if (k%100 == 0) {
            double elapsed = std::chrono::duration<double>(Clock::now()-start).count();
            if (elapsed > runtime_limit) throw std::runtime_error("Runtime limit exceeded; partial history retained");
            if (k%1000 == 0) {
                history.flush();
                std::cout << "step " << k << " tau " << as_double(R(k)*dt)
                          << " wall_seconds " << elapsed << std::endl;
            }
        }
    }
    if (remainder > R(0)) {
        Factor last(p,remainder,method); advance(last,remainder); record(end);
    }
    history.close();
    auto final = output(directory,"final_state.csv");
    final << "rho,u,v\n";
    for (int i = 0; i < p.n; ++i) final << p.rho[i] << ',' << u[i].re << ',' << v[i].re << '\n';
    final.close();
    double setup_seconds = std::chrono::duration<double>(setup_done-start).count();
    double total_seconds = std::chrono::duration<double>(Clock::now()-start).count();
    auto metadata = output(directory,"backend.json");
    metadata << "{\n  \"completed\": true,\n  \"precision\": \"" << precision_name
             << "\",\n  \"integrator\": \"" << (method == Method::radau3 ? "radau3" : "hermite")
             << "\",\n  \"method\": \""
             << (method == Method::radau3 ? "Radau IIA order 3 Pade [1/2], conjugate resolvents" : "Hermite H4 Pade [2/2], conjugate Cayley increments")
             << "\",\n"
             << "  \"n\": " << p.n << ",\n  \"bandwidth\": " << p.bw
             << ",\n  \"full_steps\": " << steps << ",\n  \"partial_step\": \"" << remainder
             << "\",\n  \"decimal_underflow_count\": " << decimal_underflow_count
             << ",\n  \"decimal_subnormal_count\": " << decimal_subnormal_count
             << ",\n  \"dt\": \"" << dt << "\",\n  \"endtime\": \"" << end
             << "\",\n  \"discarded_imaginary_max\": \"" << imaginary_max
             << "\",\n  \"state_max\": \"" << state_max
             << "\",\n  \"setup_seconds\": " << setup_seconds
             << ",\n  \"total_seconds\": " << total_seconds << "\n}\n";
    metadata.close();
    std::cout << "complete precision " << precision_name << " integrator "
              << (method == Method::radau3 ? "radau3" : "hermite") << " setup_seconds " << setup_seconds
              << " total_seconds " << total_seconds << " imaginary_max " << imaginary_max << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "hermite_banded: " << error.what() << '\n';
    return 1;
}
