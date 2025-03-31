#define grid_size_x 256
#define grid_size_y 256
#define grid_size_z 1
#define MDIMC 8
#define NDIMC 8
#define block_size_z 1
#define GEMMK 0
#define MWG 16
#define NWG 16
#define KWG 16
#define MDIMA 8
#define NDIMB 8
#define KWI 2
#define VWM 1
#define VWN 1
#define STRM 0
#define STRN 0
#define SA 0
#define SB 0
#define KREG 1
#define PRECISION 32
#define kernel_tuner 1
#line 1

// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the common defines and type-defs for the CLBlast OpenCL kernels.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
// R"(
// =================================================================================================

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this file is used outside of the CLBlast library.
#ifndef PRECISION
  #define PRECISION 32      // Data-types: half, single or double precision, complex or regular
#endif

// =================================================================================================

#ifndef CUDA
  // Enable support for half-precision
  #if PRECISION == 16
    #pragma OPENCL EXTENSION cl_khr_fp16: enable
  #endif

  // Enable support for double-precision
  #if PRECISION == 64 || PRECISION == 6464
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
  #endif
#endif

// Half-precision
#if PRECISION == 16
  typedef half real;
  typedef half2 real2;
  typedef half4 real4;
  typedef half8 real8;
  typedef half16 real16;
  #define ZERO 0
  #define ONE 1
  #define SMALLEST -1.0e14

// Single-precision
#elif PRECISION == 32
  typedef float real;
  typedef float2 real2;
  typedef float4 real4;
  typedef float8 real8;
  typedef float16 real16;
  #define ZERO 0.0f
  #define ONE 1.0f
  #define SMALLEST -1.0e37f

// Double-precision 
#elif PRECISION == 64
  typedef double real;
  typedef double2 real2;
  typedef double4 real4;
  typedef double8 real8;
  typedef double16 real16;
  #define ZERO 0.0
  #define ONE 1.0
  #define SMALLEST -1.0e37

// Complex single-precision
#elif PRECISION == 3232
  typedef float2 real;
  typedef struct cfloat2 {real x; real y;} real2;
  typedef struct cfloat4 {real x; real y; real z; real w;} real4;
  typedef struct cfloat8 {real s0; real s1; real s2; real s3;
                          real s4; real s5; real s6; real s7;} real8;
  typedef struct cfloat16 {real s0; real s1; real s2; real s3;
                           real s4; real s5; real s6; real s7;
                           real s8; real s9; real sA; real sB;
                           real sC; real sD; real sE; real sF;} real16;
  #define ZERO 0.0f
  #define ONE 1.0f
  #define SMALLEST -1.0e37f

// Complex double-precision
#elif PRECISION == 6464
  typedef double2 real;
  typedef struct cdouble2 {real x; real y;} real2;
  typedef struct cdouble4 {real x; real y; real z; real w;} real4;
  typedef struct cdouble8 {real s0; real s1; real s2; real s3;
                           real s4; real s5; real s6; real s7;} real8;
  typedef struct cdouble16 {real s0; real s1; real s2; real s3;
                            real s4; real s5; real s6; real s7;
                            real s8; real s9; real sA; real sB;
                            real sC; real sD; real sE; real sF;} real16;
  #define ZERO 0.0
  #define ONE 1.0
  #define SMALLEST -1.0e37
#endif

// Single-element version of a complex number
#if PRECISION == 3232
  typedef float singlereal;
#elif PRECISION == 6464
  typedef double singlereal;
#else
  typedef real singlereal;
#endif

// Converts a 'real argument' value to a 'real' value as passed to the kernel. Normally there is no
// conversion, but half-precision is not supported as kernel argument so it is converted from float.
#if PRECISION == 16
  typedef float real_arg;
  #define GetRealArg(x) (half)x
#else
  typedef real real_arg;
  #define GetRealArg(x) x
#endif

// Pointers to local memory objects (using a define because CUDA doesn't need them)
#ifndef LOCAL_PTR
  #define LOCAL_PTR __local
#endif

// =================================================================================================

// Don't use the non-IEEE754 compliant OpenCL built-in mad() instruction per default. For specific
// devices, this is enabled (see src/routine.cpp).
#ifndef USE_CL_MAD
  #define USE_CL_MAD 0
#endif

// By default the workgroup size requirement is enabled. For Qualcomm devices the workgroup size 
// requirement results in worse performance and is disabled (src/utilities/compile.cpp)
#ifndef RELAX_WORKGROUP_SIZE
  #define RELAX_WORKGROUP_SIZE 0
#endif

// Sets a variable to zero
#if PRECISION == 3232 || PRECISION == 6464
  #define SetToZero(a) a.x = ZERO; a.y = ZERO
#else
  #define SetToZero(a) a = ZERO
#endif

// Sets a variable to zero (only the imaginary part)
#if PRECISION == 3232 || PRECISION == 6464
  #define ImagToZero(a) a.y = ZERO
#else
  #define ImagToZero(a) 
#endif

// Sets a variable to one
#if PRECISION == 3232 || PRECISION == 6464
  #define SetToOne(a) a.x = ONE; a.y = ZERO
#else
  #define SetToOne(a) a = ONE
#endif

// Determines whether a variable is zero
#if PRECISION == 3232 || PRECISION == 6464
  #define IsZero(a) ((a.x == ZERO) && (a.y == ZERO))
#else
  #define IsZero(a) (a == ZERO)
#endif

// The absolute value (component-wise)
#if PRECISION == 3232 || PRECISION == 6464
  #define AbsoluteValue(value) value.x = fabs(value.x); value.y = fabs(value.y)
#else
  #define AbsoluteValue(value) value = fabs(value)
#endif

// Negation (component-wise)
#if PRECISION == 3232 || PRECISION == 6464
  #define Negate(value) value.x = -(value.x); value.y = -(value.y)
#else
  #define Negate(value) value = -(value)
#endif

// Adds two complex variables
#if PRECISION == 3232 || PRECISION == 6464
  #define Add(c,a,b) c.x = a.x + b.x; c.y = a.y + b.y
#else
  #define Add(c,a,b) c = a + b
#endif

// Subtracts two complex variables
#if PRECISION == 3232 || PRECISION == 6464
  #define Subtract(c,a,b) c.x = a.x - b.x; c.y = a.y - b.y
#else
  #define Subtract(c,a,b) c = a - b
#endif

// Multiply two complex variables (used in the defines below)
#if PRECISION == 3232 || PRECISION == 6464
  #define MulReal(a,b) a.x*b.x - a.y*b.y
  #define MulImag(a,b) a.x*b.y + a.y*b.x
#endif

// The scalar multiply function
#if PRECISION == 3232 || PRECISION == 6464
  #define Multiply(c,a,b) c.x = MulReal(a,b); c.y = MulImag(a,b)
#else
  #define Multiply(c,a,b) c = a * b
#endif

// The scalar multiply-add function
#if PRECISION == 3232 || PRECISION == 6464
  #define MultiplyAdd(c,a,b) c.x += MulReal(a,b); c.y += MulImag(a,b)
#else
  #if USE_CL_MAD == 1
    #define MultiplyAdd(c,a,b) c = mad(a, b, c)
  #else
    #define MultiplyAdd(c,a,b) c += a * b
  #endif
#endif

// The scalar multiply-subtract function
#if PRECISION == 3232 || PRECISION == 6464
  #define MultiplySubtract(c,a,b) c.x -= MulReal(a,b); c.y -= MulImag(a,b)
#else
  #define MultiplySubtract(c,a,b) c -= a * b
#endif

// The scalar division function: full division
#if PRECISION == 3232 || PRECISION == 6464
  #define DivideFull(c,a,b) singlereal num_x = (a.x * b.x) + (a.y * b.y); singlereal num_y = (a.y * b.x) - (a.x * b.y); singlereal denom = (b.x * b.x) + (b.y * b.y); c.x = num_x / denom; c.y = num_y / denom
#else
  #define DivideFull(c,a,b) c = a / b
#endif

// The scalar AXPBY function
#if PRECISION == 3232 || PRECISION == 6464
  #define AXPBY(e,a,b,c,d) e.x = MulReal(a,b) + MulReal(c,d); e.y = MulImag(a,b) + MulImag(c,d)
#else
  #define AXPBY(e,a,b,c,d) e = a*b + c*d
#endif

// The complex conjugate operation for complex transforms
#if PRECISION == 3232 || PRECISION == 6464
  #define COMPLEX_CONJUGATE(value) value.x = value.x; value.y = -value.y
#else
  #define COMPLEX_CONJUGATE(value) 
#endif

// =================================================================================================

// Force inlining functions or not: some compilers don't support the inline keyword
#ifdef USE_INLINE_KEYWORD
  #define INLINE_FUNC inline
#else
  #define INLINE_FUNC
#endif

// =================================================================================================

// Shuffled workgroup indices to avoid partition camping, see below. For specific devices, this is
// enabled (see src/routine.cc).
#ifndef USE_STAGGERED_INDICES
  #define USE_STAGGERED_INDICES 0
#endif

// Staggered/shuffled group indices to avoid partition camping (AMD GPUs). Formula's are taken from:
// http://docs.nvidia.com/cuda/samples/6_Advanced/transpose/doc/MatrixTranspose.pdf
// More details: https://github.com/CNugteren/CLBlast/issues/53
#if USE_STAGGERED_INDICES == 1 && GEMMK == 0
  INLINE_FUNC int GetGroupIDFlat() {
    return get_group_id(0) + get_num_groups(0) * get_group_id(1);
  }
  INLINE_FUNC int GetGroupID1() {
    return (GetGroupIDFlat()) % get_num_groups(1);
  }
  INLINE_FUNC int GetGroupID0() {
    return ((GetGroupIDFlat() / get_num_groups(1)) + GetGroupID1()) % get_num_groups(0);
  }
#else
  INLINE_FUNC int GetGroupID1() { return get_group_id(1); }
  INLINE_FUNC int GetGroupID0() { return get_group_id(0); }
#endif

// =================================================================================================

// End of the C++11 raw string literal
// )"

// =================================================================================================

// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains two optimized matrix-multiplication kernels:
// - Kernel 0: inspired by the paper by Matsumoto et al. and the tutorial on
//   http://www.cedricnugteren.nl/tutorial.php
// - Kernel 1: inspired by a Qualcomm optimized GPU kernel with 2D register tiling
//   https://developer.qualcomm.com/blog/matrix-multiply-adreno-gpus-part-2-host-code-and-kernel
// Both are fully configurable (and tunable!) using many parameters. Both kernels support
// different data-types (SGEMM/DGEMM/CGEMM/ZGEMM/HGEMM) through a pre-processor define.
//
// For kernel 0 matrices are accessed as follows:
// A: [k*M + m], with 'k' ranging from 0:K and 'm' from 0:M (m,k,m)
// B: [k*N + n], with 'k' ranging from 0:K and 'n' from 0:N (n,k,n)
// C: [n*M + m], with 'n' ranging from 0:N and 'm' from 0:M (m,n,m)
// For kernel 1, both A and C are transposed w.r.t. the above
//
// Or as an image (assuming column-major)
//       K                      
//    o-------o                 
//    |       |                 
//  N | [B^T] |                 
//    |       |                 
//    o-------o                 
//        K               N     
//    o-------o        o-----o  
//  M |  [A]  |      M | [C] |  
//    |       |        |     |  
//    o-------o        o-----o  
//                              
//
// This kernel is separated into multiple files. This is part 1 out of 4.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
// R"(

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.
#ifndef GEMMK
  #define GEMMK 0    // Kernel to choose: 0 regular, 1 with 2D register tiling
#endif
#ifndef MWG
  #define MWG 8      // Tile-size in dimension M (e.g. 64, 128)
#endif
#ifndef NWG
  #define NWG 8      // Tile-size in dimension N (e.g. 64, 128)
#endif
#ifndef KWG
  #define KWG 8      // Tile-size in dimension K (e.g. 8, 16)
#endif
#ifndef MDIMC
  #define MDIMC 8    // Threads per workgroup in M-dimension (e.g. 8, 16, 32)
#endif
#ifndef NDIMC
  #define NDIMC 8    // Threads per workgroup in N-dimension (e.g. 8, 16, 32)
#endif
#ifndef MDIMA
  #define MDIMA 8    // Re-shaped tile dimension of matrix A: KDIMA * MDIMA (kernel 0 only)
#endif
#ifndef NDIMB
  #define NDIMB 8    // Re-shaped tile dimension of matrix B: KDIMB * NDIMB (kernel 0 only)
#endif
#ifndef KWI
  #define KWI 1      // Unroll factor of the KWG loop (smaller or equal than KWG)
#endif
#ifndef VWM
  #define VWM 1      // Vector width of matrices A and C
#endif
#ifndef VWN
  #define VWN 1      // Vector width of matrix B
#endif
#ifndef STRM
  #define STRM 0     // Use strided access within a thread in the M-dimension (1) or not (0) (kernel 0 only)
#endif
#ifndef STRN
  #define STRN 0     // Use strided access within a thread in the N-dimension (1) or not (0) (kernel 0 only)
#endif
#ifndef SA
  #define SA 0       // Use local/shared memory to cache matrix A (1) or not (0) (kernel 0 only)
#endif
#ifndef SB
  #define SB 0       // Use local/shared memory to cache matrix B (1) or not (0) (kernel 0 only)
#endif
#ifndef KREG
  #define KREG 1     // Amount of register tiling in second dimension, multiple of VWN (kernel 1 only)
#endif

// Helper parameters based on the above tuning parameters
#define MWI (MWG/MDIMC)               // Work per work-item (M-dimension)
#define NWI (NWG/NDIMC)               // Work per work-item (N-dimension)
#define KDIMA ((MDIMC*NDIMC)/(MDIMA)) // Re-shaped tile dimension of matrix A: KDIMA * MDIMA
#define KDIMB ((MDIMC*NDIMC)/(NDIMB)) // Re-shaped tile dimension of matrix B: KDIMB * NDIMB
#define MWA (MWG/MDIMA)               // Amount of loads-per-thread for matrix A (M-dimension)
#define KWA (KWG/KDIMA)               // Amount of loads-per-thread for matrix A (K-dimension)
#define KWB (KWG/KDIMB)               // Amount of loads-per-thread for matrix B (K-dimension)
#define NWB (NWG/NDIMB)               // Amount of loads-per-thread for matrix B (N-dimension)

// Settings
#ifndef USE_VECTOR_MAD
  #define USE_VECTOR_MAD 0      // Unroll (0) or don't (1) unroll the vector MAD manually
#endif
#ifndef GLOBAL_MEM_FENCE
  #define GLOBAL_MEM_FENCE 0    // Global synchronisation barrier for potential better performance
#endif

#ifndef SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA
  #define SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA 0
#endif
#ifndef SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA
  #define SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA 0
#endif
#ifndef SUBGROUP_SHUFFLING_INTEL
  #define SUBGROUP_SHUFFLING_INTEL 0
#endif
#ifndef USE_SUBGROUP_SHUFFLING
  #define USE_SUBGROUP_SHUFFLING 0     // Optionally enables subgroup shuffling for Intel GPUs
#endif

// Intel subgroups (https://www.khronos.org/registry/OpenCL/extensions/intel/cl_intel_subgroups.html)
#if USE_SUBGROUP_SHUFFLING == 1 && SUBGROUP_SHUFFLING_INTEL == 1
  #pragma OPENCL EXTENSION cl_intel_subgroups: enable
  #define SUBGROUP_SIZE 8              // Assumes subgroup size is always 8 on Intel GPUs
#endif

// NVIDIA warps as subgroups using inline PTX (https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)
#if USE_SUBGROUP_SHUFFLING == 1
  #if SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA == 1 || SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1
    #define SUBGROUP_SIZE 32            // Assumes subgroup size is always 32 on NVIDIA GPUs
  #endif
#endif

#if NWI != SUBGROUP_SIZE || MDIMC < SUBGROUP_SIZE
  #undef USE_SUBGROUP_SHUFFLING
  #define USE_SUBGROUP_SHUFFLING 0     // Disables subgroups in case the assumptions don't hold
#endif

// =================================================================================================

// Data-widths in dimension M
#if VWM == 1
    typedef real realM;
#elif VWM == 2
    typedef real2 realM;
#elif VWM == 4
    typedef real4 realM;
#elif VWM == 8
    typedef real8 realM;
#elif VWM == 16
    typedef real16 realM;
#endif

// Data-widths in dimension N
#if VWN == 1
    typedef real realN;
#elif VWN == 2
    typedef real2 realN;
#elif VWN == 4
    typedef real4 realN;
#elif VWN == 8
    typedef real8 realN;
#elif VWN == 16
    typedef real16 realN;
#endif

// =================================================================================================

// Initializes the accumulation registers to zero
INLINE_FUNC realM InitAccRegisters() {
  realM result;
  #if VWM == 1
    SetToZero(result);
  #elif VWM == 2
    SetToZero(result.x);
    SetToZero(result.y);
  #elif VWM == 4
    SetToZero(result.x);
    SetToZero(result.y);
    SetToZero(result.z);
    SetToZero(result.w);
  #elif VWM == 8
    SetToZero(result.s0);
    SetToZero(result.s1);
    SetToZero(result.s2);
    SetToZero(result.s3);
    SetToZero(result.s4);
    SetToZero(result.s5);
    SetToZero(result.s6);
    SetToZero(result.s7);
  #elif VWM == 16
    SetToZero(result.s0);
    SetToZero(result.s1);
    SetToZero(result.s2);
    SetToZero(result.s3);
    SetToZero(result.s4);
    SetToZero(result.s5);
    SetToZero(result.s6);
    SetToZero(result.s7);
    SetToZero(result.s8);
    SetToZero(result.s9);
    SetToZero(result.sA);
    SetToZero(result.sB);
    SetToZero(result.sC);
    SetToZero(result.sD);
    SetToZero(result.sE);
    SetToZero(result.sF);
  #endif
  return result;
}

// =================================================================================================

// Caches global off-chip memory into local (shared) memory on-chip. This function is specific for
// caching the A input matrix.
#if SA == 1
INLINE_FUNC void GlobalToLocalA(const __global realM* restrict agm, LOCAL_PTR realM* alm,
                                const int kSizeM, const int tid, const int kwg) {
  const int la0 = tid % MDIMA;
  const int la1 = tid / MDIMA;
  #pragma unroll
  for (int _mia = 0; _mia < MWA/VWM; _mia += 1) {
    #pragma unroll
    for (int _kia = 0; _kia < KWA; _kia += 1) {

      // Computes the indices based on strided/non-strided access
      #if STRM == 0
        int mg = _mia + la0*(MWA/VWM);
      #elif STRM == 1
        int mg = la0 + _mia*MDIMA;
      #endif

      // Computes the indices for the global memory
      int kg = _kia + la1*KWA;
      int idm = mg + GetGroupID0() * (MWG/VWM);
      int idk = kg + kwg;

      // Loads the data from global memory (not transposed) into the local memory
      alm[kg*(MWG/VWM) + mg] = agm[idk*(kSizeM/VWM) + idm];
    }
  }
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
INLINE_FUNC void GlobalToLocalB(const __global realN* restrict bgm, LOCAL_PTR realN* blm,
                                const int kSizeN, const int tid, const int kwg) {
  const int lb0 = tid % NDIMB;
  const int lb1 = tid / NDIMB;
  #pragma unroll
  for (int _kib = 0; _kib < KWB; _kib += 1) {
    #pragma unroll
    for (int _nib = 0; _nib < NWB/VWN; _nib += 1) {

      // Computes the indices based on strided/non-strided access
      #if STRN == 0
        int ng = _nib + lb0*(NWB/VWN);
      #elif STRN == 1
        int ng = lb0 + _nib*NDIMB;
      #endif

      // Computes the indices for the global memory
      int kg = _kib + lb1*KWB;
      int idn = ng + GetGroupID1() * (NWG/VWN);
      int idk = kg + kwg;

      // Loads the data from global memory (transposed) into the local memory
      blm[kg*(NWG/VWN) + ng] = bgm[idk*(kSizeN/VWN) + idn];
    }
  }
}
#endif

// =================================================================================================

// Caches global off-chip memory directly into per-thread private memory (registers). This function
// is specific for caching the A input matrix.
#if SA == 0 && GEMMK == 0
INLINE_FUNC realM GlobalToPrivateA(const __global realM* restrict agm, const int _mi,
                                   const int kSizeM, const int idk, const int kwg) {
  // Computes the indices based on strided/non-strided access
  #if STRM == 0
    int mg = _mi + get_local_id(0)*(MWI/VWM);
  #elif STRM == 1
    int mg = get_local_id(0) + _mi*MDIMC;
  #endif

  // Computes the indices for the global memory
  int idm = mg + GetGroupID0() * (MWG/VWM);

  // Loads the data from global memory (not transposed) and stores into registers
  return agm[idk*(kSizeM/VWM) + idm];
}
#endif

// Same as above, but now for the B input matrix
#if SB == 0 && GEMMK == 0
INLINE_FUNC realN GlobalToPrivateB(const __global realN* restrict bgm, const int _ni,
                                   const int kSizeN, const int idk) {
  // Computes the indices based on strided/non-strided access
  #if STRN == 0
    int ng = _ni + get_local_id(1)*(NWI/VWN);
  #elif STRN == 1
    int ng = get_local_id(1) + _ni*NDIMC;
  #endif

  // Computes the indices for the global memory
  int idn = ng + GetGroupID1() * (NWG/VWN);

  // Loads the data from global memory (transposed) and stores into registers
  return bgm[idk*(kSizeN/VWN) + idn];
}
#endif

// =================================================================================================
#if GEMMK == 1

// Caches global off-chip memory directly into per-thread private memory (registers). This function
// is specific for caching the A input matrix for kernel 1.
INLINE_FUNC realN GlobalToPrivateA2D(const __global real* restrict a_ptr, const int tid_y, const int _ni,
                                     const int kSizeK, const int idk, const int _ki) {
  #if PRECISION == 3232 || PRECISION == 6464
    const int a_index = (tid_y * NWI + _ni) * (kSizeK / VWN) + idk / VWN + _ki;
    const __global realN* restrict agm = (const __global realN* restrict) a_ptr;
    return agm[a_index];
  #else
    const int a_index = (tid_y * NWI + _ni) * kSizeK + idk + _ki * VWN;
    #if VWN == 1
      return a_ptr[a_index];
    #elif VWN == 2
      return vload2(0, a_ptr + a_index);
    #elif VWN == 4
      return vload4(0, a_ptr + a_index);
    #elif VWN == 8
      return vload8(0, a_ptr + a_index);
    #elif VWN == 16
      return vload16(0, a_ptr + a_index);
    #endif
  #endif
}

// Same as above, but now for the B input matrix
INLINE_FUNC realM GlobalToPrivateB2D(const __global real* restrict b_ptr, const int tid_x, const int _mi,
                                     const int kSizeN, const int idk, const int _ki) {
  #if PRECISION == 3232 || PRECISION == 6464
    const int b_index = (idk + _ki) * (kSizeN / VWM) + tid_x * (MWI / VWM) + _mi;
    const __global realM* restrict bgm = (const __global realM* restrict) b_ptr;
    return bgm[b_index];
  #else
    const int b_index = (idk + _ki) * kSizeN + tid_x * MWI + _mi * VWM;
    #if VWM == 1
      return b_ptr[b_index];
    #elif VWM == 2
      return vload2(0, b_ptr + b_index);
    #elif VWM == 4
      return vload4(0, b_ptr + b_index);
    #elif VWM == 8
      return vload8(0, b_ptr + b_index);
    #elif VWM == 16
      return vload16(0, b_ptr + b_index);
    #endif
  #endif
}

#endif
// =================================================================================================

// Caches on-chip local memory into per-thread private memory (registers). This function is specific
// for caching the A input matrix.
#if SA == 1
INLINE_FUNC realM LocalToPrivateA(LOCAL_PTR realM* alm, const int _mi, const int kg) {
  #if STRM == 0
    int mg = _mi + get_local_id(0)*(MWI/VWM);
  #elif STRM == 1
    int mg = get_local_id(0) + _mi*MDIMC;
  #endif
  return alm[kg*(MWG/VWM) + mg];
}
#endif

// Same as above, but now for the B input matrix
#if SB == 1
INLINE_FUNC realN LocalToPrivateB(LOCAL_PTR realN* blm, const int _ni, const int kg) {
  #if STRN == 0
    int ng = _ni + get_local_id(1)*(NWI/VWN);
  #elif STRN == 1
    int ng = get_local_id(1) + _ni*NDIMC;
  #endif
  return blm[kg*(NWG/VWN) + ng];
}
#endif

// )"
// End of the C++11 raw string literal

// =================================================================================================

// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 2 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
// R"(

// The vectorised multiply-add function
INLINE_FUNC realM MultiplyAddVector(realM cvec, const realM avec, const real bval) {
  #if USE_VECTOR_MAD == 1
    cvec += avec * bval;
  #else
    #if VWM == 1
      MultiplyAdd(cvec,    avec,    bval);
    #elif VWM == 2
      MultiplyAdd(cvec.x , avec.x,  bval);
      MultiplyAdd(cvec.y , avec.y,  bval);
    #elif VWM == 4
      MultiplyAdd(cvec.x , avec.x,  bval);
      MultiplyAdd(cvec.y , avec.y,  bval);
      MultiplyAdd(cvec.z , avec.z,  bval);
      MultiplyAdd(cvec.w , avec.w,  bval);
    #elif VWM == 8
      MultiplyAdd(cvec.s0, avec.s0, bval);
      MultiplyAdd(cvec.s1, avec.s1, bval);
      MultiplyAdd(cvec.s2, avec.s2, bval);
      MultiplyAdd(cvec.s3, avec.s3, bval);
      MultiplyAdd(cvec.s4, avec.s4, bval);
      MultiplyAdd(cvec.s5, avec.s5, bval);
      MultiplyAdd(cvec.s6, avec.s6, bval);
      MultiplyAdd(cvec.s7, avec.s7, bval);
    #elif VWM == 16
      MultiplyAdd(cvec.s0, avec.s0, bval);
      MultiplyAdd(cvec.s1, avec.s1, bval);
      MultiplyAdd(cvec.s2, avec.s2, bval);
      MultiplyAdd(cvec.s3, avec.s3, bval);
      MultiplyAdd(cvec.s4, avec.s4, bval);
      MultiplyAdd(cvec.s5, avec.s5, bval);
      MultiplyAdd(cvec.s6, avec.s6, bval);
      MultiplyAdd(cvec.s7, avec.s7, bval);
      MultiplyAdd(cvec.s8, avec.s8, bval);
      MultiplyAdd(cvec.s9, avec.s9, bval);
      MultiplyAdd(cvec.sA, avec.sA, bval);
      MultiplyAdd(cvec.sB, avec.sB, bval);
      MultiplyAdd(cvec.sC, avec.sC, bval);
      MultiplyAdd(cvec.sD, avec.sD, bval);
      MultiplyAdd(cvec.sE, avec.sE, bval);
      MultiplyAdd(cvec.sF, avec.sF, bval);
    #endif
  #endif
  return cvec;
}

// =================================================================================================

// Merges the results in Cpm with the global array in Cgm. This also performs the multiplication
// with the constants: Cgm = alpha*A*B + beta*Cgm = alpha*Cpm + beta*Cgm
INLINE_FUNC void StoreResults(__global realM* cgm, realM c_value, const int _mi, const int _ni,
                              const int kSizeM, const real alpha, const real beta) {
  #if STRM == 0
    int mg = _mi + get_local_id(0)*(MWI/VWM);
  #elif STRM == 1
    int mg = get_local_id(0) + _mi*MDIMC;
  #endif
  #if STRN == 0
    int ng = _ni + get_local_id(1)*NWI;
  #elif STRN == 1
    int ng = _ni%VWN + get_local_id(1)*VWN + (_ni/VWN)*VWN*NDIMC;
  #endif
  int idm = mg + GetGroupID0() * (MWG/VWM);
  int idn = ng + GetGroupID1() * NWG;
  int index = idn*(kSizeM/VWM) + idm;

  realM result;
  realM xval = c_value;

  // The final multiplication with alpha (in case beta == 0)
  if (IsZero(beta)) {
    #if VWM == 1
      Multiply(result, alpha, xval);
    #elif VWM == 2
      Multiply(result.x, alpha, xval.x);
      Multiply(result.y, alpha, xval.y);
    #elif VWM == 4
      Multiply(result.x, alpha, xval.x);
      Multiply(result.y, alpha, xval.y);
      Multiply(result.z, alpha, xval.z);
      Multiply(result.w, alpha, xval.w);
    #elif VWM == 8
      Multiply(result.s0, alpha, xval.s0);
      Multiply(result.s1, alpha, xval.s1);
      Multiply(result.s2, alpha, xval.s2);
      Multiply(result.s3, alpha, xval.s3);
      Multiply(result.s4, alpha, xval.s4);
      Multiply(result.s5, alpha, xval.s5);
      Multiply(result.s6, alpha, xval.s6);
      Multiply(result.s7, alpha, xval.s7);
    #elif VWM == 16
      Multiply(result.s0, alpha, xval.s0);
      Multiply(result.s1, alpha, xval.s1);
      Multiply(result.s2, alpha, xval.s2);
      Multiply(result.s3, alpha, xval.s3);
      Multiply(result.s4, alpha, xval.s4);
      Multiply(result.s5, alpha, xval.s5);
      Multiply(result.s6, alpha, xval.s6);
      Multiply(result.s7, alpha, xval.s7);
      Multiply(result.s8, alpha, xval.s8);
      Multiply(result.s9, alpha, xval.s9);
      Multiply(result.sA, alpha, xval.sA);
      Multiply(result.sB, alpha, xval.sB);
      Multiply(result.sC, alpha, xval.sC);
      Multiply(result.sD, alpha, xval.sD);
      Multiply(result.sE, alpha, xval.sE);
      Multiply(result.sF, alpha, xval.sF);
    #endif
  }

  // The final multiplication with alpha and the addition with beta*C
  else {
    realM yval = cgm[index];
    #if VWM == 1
      AXPBY(result, alpha, xval, beta, yval);
    #elif VWM == 2
      AXPBY(result.x, alpha, xval.x, beta, yval.x);
      AXPBY(result.y, alpha, xval.y, beta, yval.y);
    #elif VWM == 4
      AXPBY(result.x, alpha, xval.x, beta, yval.x);
      AXPBY(result.y, alpha, xval.y, beta, yval.y);
      AXPBY(result.z, alpha, xval.z, beta, yval.z);
      AXPBY(result.w, alpha, xval.w, beta, yval.w);
    #elif VWM == 8
      AXPBY(result.s0, alpha, xval.s0, beta, yval.s0);
      AXPBY(result.s1, alpha, xval.s1, beta, yval.s1);
      AXPBY(result.s2, alpha, xval.s2, beta, yval.s2);
      AXPBY(result.s3, alpha, xval.s3, beta, yval.s3);
      AXPBY(result.s4, alpha, xval.s4, beta, yval.s4);
      AXPBY(result.s5, alpha, xval.s5, beta, yval.s5);
      AXPBY(result.s6, alpha, xval.s6, beta, yval.s6);
      AXPBY(result.s7, alpha, xval.s7, beta, yval.s7);
    #elif VWM == 16
      AXPBY(result.s0, alpha, xval.s0, beta, yval.s0);
      AXPBY(result.s1, alpha, xval.s1, beta, yval.s1);
      AXPBY(result.s2, alpha, xval.s2, beta, yval.s2);
      AXPBY(result.s3, alpha, xval.s3, beta, yval.s3);
      AXPBY(result.s4, alpha, xval.s4, beta, yval.s4);
      AXPBY(result.s5, alpha, xval.s5, beta, yval.s5);
      AXPBY(result.s6, alpha, xval.s6, beta, yval.s6);
      AXPBY(result.s7, alpha, xval.s7, beta, yval.s7);
      AXPBY(result.s8, alpha, xval.s8, beta, yval.s8);
      AXPBY(result.s9, alpha, xval.s9, beta, yval.s9);
      AXPBY(result.sA, alpha, xval.sA, beta, yval.sA);
      AXPBY(result.sB, alpha, xval.sB, beta, yval.sB);
      AXPBY(result.sC, alpha, xval.sC, beta, yval.sC);
      AXPBY(result.sD, alpha, xval.sD, beta, yval.sD);
      AXPBY(result.sE, alpha, xval.sE, beta, yval.sE);
      AXPBY(result.sF, alpha, xval.sF, beta, yval.sF);
    #endif
  }
  cgm[index] = result;
}

// )"
// End of the C++11 raw string literal

// =================================================================================================

// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 3 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
// R"(

// A common interface for subgroup functions

#if USE_SUBGROUP_SHUFFLING == 1

INLINE_FUNC int clblast_get_sub_group_local_id() {

  // Intel extension 
  #if SUBGROUP_SHUFFLING_INTEL == 1
  return get_sub_group_local_id();
  
  // Nvidia inline PTX
  #elif SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA == 1 || SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1
  int ret;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(ret) );
  return ret;
  #endif 
}

INLINE_FUNC realN clblast_sub_group_shuffle(realN reg, int src) {

  // Intel extension 
  #if SUBGROUP_SHUFFLING_INTEL == 1
  return intel_sub_group_shuffle(reg, src);
  
  // Nvidia inline PTX
  // Volta and later requires .sync shuffle instructions with an extra mask arg
  #elif SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA == 1 || SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1
  realN ret;
    #if SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA == 1
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(ret): "f"(reg), "r"(src));
    #else
    asm volatile("shfl.idx.b32 %0, %1, %2, 0x1f;" : "=f"(ret): "f"(reg), "r"(src));
    #endif
  return ret;
  #endif
}
#endif

// Main body of the matrix-multiplication algorithm. It calls various (inlined) functions.
INLINE_FUNC void XgemmBody(const int kSizeM, const int kSizeN, const int kSizeK,
                           const __global realM* restrict agm, const __global realN* restrict bgm,
                           __global realM* cgm, const real alpha, const real beta
                           #if SA == 1 && SB == 1
                             , LOCAL_PTR realM* alm, LOCAL_PTR realN* blm
                           #elif SA == 1
                             , LOCAL_PTR realM* alm
                           #elif SB == 1
                             , LOCAL_PTR realN* blm
                           #endif
                           ) {

  // Allocates workitem-private memory (registers)
  #if GEMMK == 0
    
    realM apm[MWI/VWM]; // MWI * 1
    
    realN bpm[NWI/VWN]; // 1 * NWI
  #elif GEMMK == 1
    #if USE_SUBGROUP_SHUFFLING == 1
      
      realN apm[KREG/VWN]; // KREG (subgroup shuffling in NWI dimension)
    #else
      
      realN apm[NWI*(KREG/VWN)]; // NWI * KREG
    #endif
    
    realM bpm[KREG*(MWI/VWM)]; // KREG * MWI
  #endif
  
  realM cpm[NWI*(MWI/VWM)]; // NWI * MWI

  #if GEMMK == 1
    const __global real* restrict a_ptr = (const __global real* restrict) &agm[0];
    const __global real* restrict b_ptr = (const __global real* restrict) &bgm[0];
    const int tid_x = get_local_id(0) + MDIMC * GetGroupID0();
    const int tid_y = get_local_id(1) + NDIMC * GetGroupID1();
  #endif

  // Combined thread identifier (volatile to disable caching)
  #if SA == 1 || SB == 1
    volatile int tid = get_local_id(0) + MDIMC*get_local_id(1);
  #endif

  // Initializes the accumulation registers
  #pragma unroll
  for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
    #pragma unroll
    for (int _ni = 0; _ni < NWI; _ni += 1) {
      cpm[_ni * (MWI/VWM) + _mi] = InitAccRegisters();
    }
  }

  // Loops over all workgroup tiles
  for (int kwg = 0; kwg < kSizeK; kwg += KWG * KREG) {

    // Loads data: off-chip --> local (matrix A)
    #if SA == 1
      GlobalToLocalA(agm, alm, kSizeM, tid, kwg);
    #endif
    // Loads data: off-chip --> local (matrix B)
    #if SB == 1
      GlobalToLocalB(bgm, blm, kSizeN, tid, kwg);
    #endif
    #if SA == 1 || SB == 1
      barrier(CLK_LOCAL_MEM_FENCE);
    #endif

    // Loops over all workitem tiles, unrolled by a factor KWI
    for (int pwi = 0; pwi < KWG * KREG; pwi += KWI * KREG) {
      #pragma unroll
      for (int _pit = 0; _pit < KWI*KREG; _pit += KREG) {
        #if SA == 0 || SB == 0
          int idk = kwg + pwi + _pit;
        #endif
        #if SA == 1 || SB == 1
          int kg = pwi + _pit;
        #endif

        // Loads matrix A (kernel 0) or matrix B (kernel 1)
        #pragma unroll
        for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
          // Loads data: local --> private (matrix A)
          #if GEMMK == 0 && SA == 1
            apm[_mi] = LocalToPrivateA(alm, _mi, kg);
          // Loads data: off-chip --> private (matrix A)
          #elif GEMMK == 0 && SA == 0
            apm[_mi] = GlobalToPrivateA(agm, _mi, kSizeM, idk, kwg);
          // Loads data: 2D global --> 2D private (matrix B)
          #elif GEMMK == 1
            #pragma unroll
            for (int _ki = 0; _ki < KREG; _ki += 1) {
              bpm[_ki * (MWI/VWM) + _mi] = GlobalToPrivateB2D(b_ptr, tid_x, _mi, kSizeN, idk, _ki);
            }
          #endif
        }

        // Loads matrix B (kernel 0) or matrix A (kernel 1)
        #if GEMMK == 0
          #pragma unroll
          for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
            // Loads data: local --> private (matrix B)
            #if SB == 1
              bpm[_ni] = LocalToPrivateB(blm, _ni, kg);
            // Loads data: off-chip --> private (matrix B)
            #else
              bpm[_ni] = GlobalToPrivateB(bgm, _ni, kSizeN, idk);
            #endif
          }
        #elif GEMMK == 1
          // Loads data: 2D global --> 2D private (matrix A). Partly, shuffled later among subgroups
          #if USE_SUBGROUP_SHUFFLING == 1
            const int _ni = clblast_get_sub_group_local_id();
            #pragma unroll
            for (int _ki = 0; _ki < KREG/VWN; _ki += 1) {
              apm[_ki] = GlobalToPrivateA2D(a_ptr, tid_y, _ni, kSizeK, idk, _ki);
            }
          // Loads data: 2D global --> 2D private (matrix A)
          #else
            #pragma unroll
            for (int _ni = 0; _ni < NWI; _ni += 1) {
              #pragma unroll
              for (int _ki = 0; _ki < KREG/VWN; _ki += 1) {
                apm[_ni * (KREG/VWN) + _ki] = GlobalToPrivateA2D(a_ptr, tid_y, _ni, kSizeK, idk, _ki);
              }
            }
          #endif
        #endif

        // Performs the accumulation (Cpm += Apm * Bpm)
        #if GEMMK == 0
          #pragma unroll
          for (int _ni = 0; _ni < NWI/VWN; _ni += 1) {
            #pragma unroll
            for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
              const realM aval = apm[_mi];
              #if VWN == 1
                cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni]);
              #elif VWN == 2
                cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni].x);
                cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[_ni].y);
              #elif VWN == 4
                cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni].x);
                cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[_ni].y);
                cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi], aval, bpm[_ni].z);
                cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi], aval, bpm[_ni].w);
              #elif VWN == 8
                cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0)*(MWI/VWM) + _mi], aval, bpm[_ni].s0);
                cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1)*(MWI/VWM) + _mi], aval, bpm[_ni].s1);
                cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2)*(MWI/VWM) + _mi], aval, bpm[_ni].s2);
                cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3)*(MWI/VWM) + _mi], aval, bpm[_ni].s3);
                cpm[(_ni*VWN + 4)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 4)*(MWI/VWM) + _mi], aval, bpm[_ni].s4);
                cpm[(_ni*VWN + 5)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 5)*(MWI/VWM) + _mi], aval, bpm[_ni].s5);
                cpm[(_ni*VWN + 6)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 6)*(MWI/VWM) + _mi], aval, bpm[_ni].s6);
                cpm[(_ni*VWN + 7)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 7)*(MWI/VWM) + _mi], aval, bpm[_ni].s7);
              #elif VWN == 16
                cpm[(_ni*VWN + 0 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 0 )*(MWI/VWM) + _mi], aval, bpm[_ni].s0);
                cpm[(_ni*VWN + 1 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 1 )*(MWI/VWM) + _mi], aval, bpm[_ni].s1);
                cpm[(_ni*VWN + 2 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 2 )*(MWI/VWM) + _mi], aval, bpm[_ni].s2);
                cpm[(_ni*VWN + 3 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 3 )*(MWI/VWM) + _mi], aval, bpm[_ni].s3);
                cpm[(_ni*VWN + 4 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 4 )*(MWI/VWM) + _mi], aval, bpm[_ni].s4);
                cpm[(_ni*VWN + 5 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 5 )*(MWI/VWM) + _mi], aval, bpm[_ni].s5);
                cpm[(_ni*VWN + 6 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 6 )*(MWI/VWM) + _mi], aval, bpm[_ni].s6);
                cpm[(_ni*VWN + 7 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 7 )*(MWI/VWM) + _mi], aval, bpm[_ni].s7);
                cpm[(_ni*VWN + 8 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 8 )*(MWI/VWM) + _mi], aval, bpm[_ni].s8);
                cpm[(_ni*VWN + 9 )*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 9 )*(MWI/VWM) + _mi], aval, bpm[_ni].s9);
                cpm[(_ni*VWN + 10)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 10)*(MWI/VWM) + _mi], aval, bpm[_ni].sA);
                cpm[(_ni*VWN + 11)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 11)*(MWI/VWM) + _mi], aval, bpm[_ni].sB);
                cpm[(_ni*VWN + 12)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 12)*(MWI/VWM) + _mi], aval, bpm[_ni].sC);
                cpm[(_ni*VWN + 13)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 13)*(MWI/VWM) + _mi], aval, bpm[_ni].sD);
                cpm[(_ni*VWN + 14)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 14)*(MWI/VWM) + _mi], aval, bpm[_ni].sE);
                cpm[(_ni*VWN + 15)*(MWI/VWM) + _mi] = MultiplyAddVector(cpm[(_ni*VWN + 15)*(MWI/VWM) + _mi], aval, bpm[_ni].sF);
              #endif
            }
          }
        #elif GEMMK == 1
          #pragma unroll
          for (int _ni = 0; _ni < NWI; _ni += 1) {
            #pragma unroll
            for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
              #pragma unroll
              for (int _ki = 0; _ki < KREG/VWN; _ki += 1) {
                #if USE_SUBGROUP_SHUFFLING == 1
                  const realN aval = clblast_sub_group_shuffle(apm[_ki], _ni);
                #else
                  const realN aval = apm[_ni * (KREG/VWN) + _ki];
                #endif
                #if VWN == 1
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 0) * (MWI/VWM) + _mi], aval);
                #elif VWN == 2
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 0) * (MWI/VWM) + _mi], aval.x);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 1) * (MWI/VWM) + _mi], aval.y);
                #elif VWN == 4
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 0) * (MWI/VWM) + _mi], aval.x);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 1) * (MWI/VWM) + _mi], aval.y);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 2) * (MWI/VWM) + _mi], aval.z);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 3) * (MWI/VWM) + _mi], aval.w);
                #elif VWN == 8
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 0) * (MWI/VWM) + _mi], aval.s0);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 1) * (MWI/VWM) + _mi], aval.s1);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 2) * (MWI/VWM) + _mi], aval.s2);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 3) * (MWI/VWM) + _mi], aval.s3);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 4) * (MWI/VWM) + _mi], aval.s4);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 5) * (MWI/VWM) + _mi], aval.s5);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 6) * (MWI/VWM) + _mi], aval.s6);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 7) * (MWI/VWM) + _mi], aval.s7);
                #elif VWN == 16
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 0 ) * (MWI/VWM) + _mi], aval.s0);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 1 ) * (MWI/VWM) + _mi], aval.s1);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 2 ) * (MWI/VWM) + _mi], aval.s2);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 3 ) * (MWI/VWM) + _mi], aval.s3);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 4 ) * (MWI/VWM) + _mi], aval.s4);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 5 ) * (MWI/VWM) + _mi], aval.s5);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 6 ) * (MWI/VWM) + _mi], aval.s6);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 7 ) * (MWI/VWM) + _mi], aval.s7);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 8 ) * (MWI/VWM) + _mi], aval.s8);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 9 ) * (MWI/VWM) + _mi], aval.s9);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 10) * (MWI/VWM) + _mi], aval.sA);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 11) * (MWI/VWM) + _mi], aval.sB);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 12) * (MWI/VWM) + _mi], aval.sC);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 13) * (MWI/VWM) + _mi], aval.sD);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 14) * (MWI/VWM) + _mi], aval.sE);
                  cpm[_ni * (MWI/VWM) + _mi] = MultiplyAddVector(cpm[_ni * (MWI/VWM) + _mi], bpm[(VWN * _ki + 15) * (MWI/VWM) + _mi], aval.sF);
                #endif
              }
            }
          }
        #endif

      }
    }
    #if SA == 1 || SB == 1
      barrier(CLK_LOCAL_MEM_FENCE);
    #endif
  }
  #if GLOBAL_MEM_FENCE == 1
    barrier(CLK_GLOBAL_MEM_FENCE);
  #endif

  // Stores an MWG * NWG tile of results and performs the multiplication with alpha and beta
  #if GEMMK == 0
    const int cld = kSizeM;
  #elif GEMMK == 1
    const int cld = kSizeN;
  #endif
  #pragma unroll
  for (int _ni = 0; _ni < NWI; _ni += 1) {
    #pragma unroll
    for (int _mi = 0; _mi < MWI/VWM; _mi += 1) {
      StoreResults(cgm, cpm[_ni * (MWI/VWM) + _mi], _mi, _ni, cld, alpha, beta);
    }
  }
}

// )"
// End of the C++11 raw string literal

// =================================================================================================

// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This is part 4 of 4 of the GEMM kernel. See part 1 for more information.
//
// =================================================================================================

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
// R"(

// The upper-triangular and lower-triangular kernels are only used in special cases
#if defined(ROUTINE_SYRK) || defined(ROUTINE_HERK) || defined(ROUTINE_SYR2K) || defined(ROUTINE_HER2K)

// Main entry point of the kernel. This is the upper-triangular version.
void XgemmUpper(const int kSizeN, const int kSizeK,
                const real_arg arg_alpha,
                const real_arg arg_beta,
                const __global realM* restrict agm,
                const __global realN* restrict bgm,
                __global realM* cgm) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  // Skip these threads if they do not contain threads contributing to the upper-triangle
  if ((GetGroupID1() + 1)*NWG < GetGroupID0()*MWG) {
    return;
  }

  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realN blm[KWG * NWG/VWN];
  #endif

  // Computes the matrix-multiplication and stores the result in global memory
  #if SA == 1 && SB == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm, blm);
  #elif SA == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm);
  #elif SB == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, blm);
  #else
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta);
  #endif
}

// Main entry point of the kernel. This is the lower-triangular version.
void XgemmLower(const int kSizeN, const int kSizeK,
                const real_arg arg_alpha,
                const real_arg arg_beta,
                const __global realM* restrict agm,
                const __global realN* restrict bgm,
                __global realM* cgm) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  // Skip these threads if they do not contain threads contributing to the lower-triangle
  if (GetGroupID1()*NWG > (GetGroupID0() + 1)*MWG) {
    return;
  }

  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realN blm[KWG * NWG/VWN];
  #endif

  // Computes the matrix-multiplication and stores the result in global memory
  #if SA == 1 && SB == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm, blm);
  #elif SA == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm);
  #elif SB == 1
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, blm);
  #else
    XgemmBody(kSizeN, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta);
  #endif
}

// =================================================================================================
// If not using a triangular version, include the regular kernel
#else

// Main entry point of the kernel. This is the regular full version.
__global__ void Xgemm(const int kSizeM, const int kSizeN, const int kSizeK,
           const real_arg arg_alpha,
           const real_arg arg_beta,
           const __global realM* restrict agm,
           const __global realN* restrict bgm,
           __global realM* cgm,
           const int b_offset, const int c_offset) {
  const real alpha = GetRealArg(arg_alpha);
  const real beta = GetRealArg(arg_beta);

  // Adds the offsets (in case of use of a single temporary buffer for A, B, and C)
  bgm = &bgm[b_offset];
  cgm = &cgm[c_offset];

  // Allocates workgroup-private memory (local memory)
  #if SA == 1
    __local realM alm[KWG * MWG/VWM];
  #endif
  #if SB == 1
    __local realN blm[KWG * NWG/VWN];
  #endif

  // Computes the matrix-multiplication and stores the result in global memory
  #if SA == 1 && SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm, blm);
  #elif SA == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, alm);
  #elif SB == 1
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta, blm);
  #else
    XgemmBody(kSizeM, kSizeN, kSizeK, agm, bgm, cgm, alpha, beta);
  #endif
}

#endif

// )"
// End of the C++11 raw string literal

// =================================================================================================
