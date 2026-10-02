#ifndef _QD_QD_CONFIG_H
#define _QD_QD_CONFIG_H
#define QD_API
#define QD_INLINE 1
#define QD_HAVE_STD 1
#define QD_IEEE_ADD 1
#define QD_FMA(a,b,c) std::fma((a),(b),(c))
#define QD_FMS(a,b,c) std::fma((a),(b),-(c))
#define QD_ISFINITE(x) std::isfinite(x)
#define QD_ISINF(x) std::isinf(x)
#define QD_ISNAN(x) std::isnan(x)
#endif
