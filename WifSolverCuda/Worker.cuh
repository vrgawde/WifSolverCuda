
#include <stdint.h>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lib/hash/sha256.cu"
#include "lib/Math.cuh"


__global__ void kernelUncompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks);
__global__ void kernelCompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks);
__global__ void kernelUncompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks, const uint32_t checksum);
__global__ void kernelCompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks, const uint32_t checksum);
__global__ void resultCollector(bool* buffResult, uint64_t* buffCombinedResult, const uint64_t threadsInBlockNumberOfChecks);

__global__ void kernelCompressed(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks);
__global__ void kernelCompressed(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks, const uint32_t checksum);
__global__ void kernelUncompressed(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks);
__global__ void kernelUncompressed(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks, const uint32_t checksum);

__device__ bool _checksumDoubleSha256CheckUncompressed(unsigned int checksum, beu32* d_hash, uint64_t* _start);
__device__ bool _checksumDoubleSha256CheckCompressed(unsigned int checksum, beu32* d_hash, uint64_t* _start);

__device__ bool _checksumDoubleSha256(unsigned int checksum, beu32* d_hash);

__device__ void _add(uint64_t* C, uint64_t* A);
__device__ void _load(uint64_t* C, uint64_t* A);

__device__ void IMult(uint64_t* r, uint64_t* a, int64_t b); 

__device__ void initShared();
__device__ __inline__ void summaryShared(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag);

cudaError_t loadStride(uint64_t* stride);


// === Mini-key kernels (GPU) ================================================

// Scans a chunk of mini-keys starting from base-58 digit array `startDigits`.
// If it finds a valid mini-key (SHA256(mini + '?')[0] == 0x00), it writes the
// derived priv32 (SHA256(mini)) to `unifiedKey[32]`, sets *isResultFlag = true,
// and returns immediately (one winning key per launch).
__global__ void kernelMini(
    const int gpuIx,
    uint8_t* unifiedKey,      // [out] priv32 if found
    bool* isResultFlag,    // [out] set to true if a key is found
    uint8_t* __restrict__ startDigits, // [in] 22 base-58 indexes (0..57), 'S' is implied
    const int threadNumberOfChecks
);

// Collector used by other modes; leave as-is if already present.
__global__ void resultCollector(bool* buffResult, uint64_t* buffCombinedResult, const uint64_t threadsInBlockNumberOfChecks);

// SHA-256 chunk function from your lib; already included via "lib/hash/sha256.cu"
__device__ void sha256Kernel(beu32* const hash, C16(COMMA, EMPTY));

