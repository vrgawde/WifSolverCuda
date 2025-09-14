#include "Worker.cuh"
#include <cstdio>
#include <cstdint>


__device__ __constant__ uint64_t _stride[5];
__device__ __shared__ uint32_t _blockResults[4096];
__device__ __shared__ bool _blockResultFlag[1];
__device__ __constant__ char _b58Alphabet[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// Add 'add' in base-58 to a 22-digit (0..57) array IN-PLACE.
__device__ __forceinline__ void addUint64ToDigits(uint8_t* digits, uint64_t add) {
    int i = 21; // last position
    while (add && i >= 0) {
        uint64_t total = digits[i] + add;
        digits[i] = total % 58ull;
        add = total / 58ull;
        i--;
    }
}

// ++digits in base-58 (carry)
__device__ void incrementDigits(uint8_t* digits) {
    int i = 21;
    while (i >= 0) {
        if (digits[i] < 57) {
            digits[i]++;
            return;
        }
        digits[i] = 0;
        i--;
    }
}

// One-block SHA-256 for ASCII input up to 23/31 bytes, storing BE words in `hash[0..7]`.
__device__ void sha256MiniBlock(const char* input, int len, beu32* hash) {
    beu32 w0 = ((beu32)input[0] << 24) | ((beu32)input[1] << 16) | ((beu32)input[2] << 8) | (uint8_t)input[3];
    beu32 w1 = ((beu32)input[4] << 24) | ((beu32)input[5] << 16) | ((beu32)input[6] << 8) | (uint8_t)input[7];
    beu32 w2 = ((beu32)input[8] << 24) | ((beu32)input[9] << 16) | ((beu32)input[10] << 8) | (uint8_t)input[11];
    beu32 w3 = ((beu32)input[12] << 24) | ((beu32)input[13] << 16) | ((beu32)input[14] << 8) | (uint8_t)input[15];
    beu32 w4 = ((beu32)input[16] << 24) | ((beu32)input[17] << 16) | ((beu32)input[18] << 8) | (uint8_t)input[19];

    // pad at w5 depending on len 22 vs 23 (S + 21 chars) or (S + 21 chars + '?')
    beu32 w5;
    if (len == 22) {
        w5 = ((beu32)input[20] << 24) | ((beu32)input[21] << 16) | 0x00008000;
    }
    else {
        // len == 23 (mini + '?')
        w5 = ((beu32)input[20] << 24) | ((beu32)input[21] << 16) | ((beu32)input[22] << 8) | 0x00000080;
    }

    // Remaining words are zeros; last word is bit-length (len * 8).
    sha256Kernel(hash,
        w0, w1, w2, w3, w4, w5,
        0, 0, 0, 0, 0, 0, 0, 0,      // w6..w13
        0,                           // w14
        len * 8                      // w15 = bit length
    );
}

// returns how many trailing positions [21, 21-d+1] changed (>=1)
__device__ __forceinline__ int incrementDigits_retDepth(uint8_t d[22]) {
    for (int j = 21; j >= 1; --j) {
        if (d[j] < 57) { d[j]++; return 21 - j + 1; }
        d[j] = 0;
    }
    return 21; // overflow ignored by CPU range cap
}

__device__ __forceinline__ void addUint64ToDigits_ascii(uint8_t d[22], char m[22], uint64_t add, const char* __restrict__ b58) {
    uint64_t carry = add;
    for (int j = 21; j >= 1 && carry; --j) {
        uint64_t v = (uint64_t)d[j] + (carry % 58ULL);
        d[j] = (uint8_t)(v % 58ULL);
        carry = (carry / 58ULL) + (v / 58ULL);
    }
    // build ascii once
    m[0] = 'S';
#pragma unroll
    for (int j = 1; j < 22; ++j) m[j] = b58[d[j]];
}

__device__ __forceinline__ void dbg_print_tid_tix_mini(
    uint64_t tid, uint64_t tIx, const char mini[22], int bx, int tx)
{
    // Make a null-terminated copy of mini for printf
    char miniZ[23];
#pragma unroll
    for (int i = 0; i < 22; ++i) miniZ[i] = mini[i];
    miniZ[22] = '\0';

    // Device printf supports %llu for unsigned long long
    printf("[b%d t%d] tid=%llu tIx=%llu mini=%s\n",
        bx, tx, (unsigned long long)tid, (unsigned long long)tIx, miniZ);
}


// === MINI-KEY KERNEL =======================================================
// Grid-stride over THREAD_STEPS per-thread starting from base-58 digit-array `startDigits`.
// We synthesize ASCII "S................." from digits, test SHA256(mini + '?')[0] == 0x00,
// and if valid, store priv32 = SHA256(mini) into unifiedKey and set isResultFlag.
__global__ void kernelMini(
    int gpuIx,
    uint8_t* __restrict__ unifiedKey,
    int* __restrict__ isResultFlag,
    const uint8_t* __restrict__ startDigits,
    int threadNumberOfChecks,
    unsigned long long* __restrict__ foundOffset)
{
    // load & offset
    uint8_t digits[22];
#pragma unroll
    for (int i = 0; i < 22; ++i) digits[i] = startDigits[i];

    const uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t tIx = tid * (uint64_t)threadNumberOfChecks;

    char mini[22];
    addUint64ToDigits_ascii(digits, mini, tIx, _b58Alphabet); // builds initial ASCII once

    unsigned mask = __activemask();

    for (int i = 0; i < threadNumberOfChecks; ++i) {
        // sha256(mini + '?')
        char miniQ[23];
#pragma unroll
        for (int j = 0; j < 22; ++j) miniQ[j] = mini[j];
        miniQ[22] = '?';

        beu32 h[8];
        sha256MiniBlock(miniQ, 23, h);

        // check first byte
        uint8_t b0 = (uint8_t)(h[0] >> 24); // adjust if your sha returns LE
        int local_hit = (b0 == 0);

        if (__any_sync(mask, local_hit)) {
            if (local_hit) {
                beu32 p[8];
                sha256MiniBlock(mini, 22, p);
                dbg_print_tid_tix_mini(tid, tIx, mini, blockIdx.x, threadIdx.x);
                if (atomicCAS(isResultFlag, 0, 1) == 0) {
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        unifiedKey[k * 4 + 0] = (uint8_t)(p[k] >> 24);
                        unifiedKey[k * 4 + 1] = (uint8_t)(p[k] >> 16);
                        unifiedKey[k * 4 + 2] = (uint8_t)(p[k] >> 8);
                        unifiedKey[k * 4 + 3] = (uint8_t)(p[k] >> 0);
                    }
                    *foundOffset = tIx + (uint64_t)i;
                    __threadfence_system();
                }
            }
            return;
        }

        // next candidate: update only changed tail
        int depth = incrementDigits_retDepth(digits);
#pragma unroll
        for (int j = 22 - depth; j < 22; ++j) mini[j] = _b58Alphabet[digits[j]];

        if ((i & 31) == 0 && *isResultFlag) return;
    }
}



__global__ void kernelUncompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks, const uint32_t checksum) {
    uint64_t _start[5];
    beu32 d_hash[8];

    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_start, _stride, tIx);
    _add(_start, buffRangeStart);
    for (uint64_t i = 0, resultIx = tIx; i < threadNumberOfChecks; i++, resultIx++) {
        if (_checksumDoubleSha256CheckUncompressed(checksum, d_hash, _start)) {
            buffResult[resultIx] = true;
            buffCollectorWork[0] = true;
        }
        _add(_start, _stride);        
    }
}
__global__ void kernelCompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks, const uint32_t checksum) {
    uint64_t _start[5];
    beu32 d_hash[8];

    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_start, _stride, tIx);
    _add(_start, buffRangeStart);
    for (uint64_t i = 0, resultIx = tIx; i < threadNumberOfChecks; i++, resultIx++) {
        if (((_start[0] & 0xff00000000) >> 32) != 0x01) {
            _add(_start, _stride);
            continue;
        }
        if (_checksumDoubleSha256CheckCompressed(checksum, d_hash, _start)) {
            buffResult[resultIx] = true;
            buffCollectorWork[0] = true;
        }        
        _add(_start, _stride);
    }    
}
__global__ void kernelUncompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks) {
	uint64_t _start[5];
    beu32 d_hash[8];

    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_start, _stride, tIx);
    _add(_start, buffRangeStart);
	for (uint64_t i = 0, resultIx = tIx ; i < threadNumberOfChecks; i++, resultIx++) {
        if (_checksumDoubleSha256CheckUncompressed(_start[0] & 0xffffffff, d_hash, _start)) {
            buffResult[resultIx] = true;
            buffCollectorWork[0] = true;
        }        
		_add(_start, _stride);
	}
}
__global__ void kernelCompressed(bool* buffResult, bool* buffCollectorWork, uint64_t* const __restrict__  buffRangeStart, const int threadNumberOfChecks) {
    uint64_t _start[5];
    beu32 d_hash[8];

    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_start, _stride, tIx);
    _add(_start, buffRangeStart);
    for (uint64_t i = 0, resultIx = tIx; i < threadNumberOfChecks; i++, resultIx++) {
        if (((_start[0] & 0xff00000000) >> 32) != 0x01) {
            _add(_start, _stride);
            continue;
        }
        if (_checksumDoubleSha256CheckCompressed(_start[0] & 0xffffffff, d_hash, _start)) {
            buffResult[resultIx] = true;
            buffCollectorWork[0] = true;
        }        
        _add(_start, _stride);
    }
}
__global__ void resultCollector(bool* buffResult, uint64_t* buffCombinedResult, const uint64_t threadsInBlockNumberOfChecks) {
    if (buffCombinedResult[blockIdx.x] == 0xffffffffffff) {
        return;
    }
    uint64_t starterI = 0, starter = blockIdx.x * blockDim.x * threadsInBlockNumberOfChecks;
    if (buffCombinedResult[blockIdx.x] != 0) {
        starterI = buffCombinedResult[blockIdx.x] - starter + 1;
        starter = buffCombinedResult[blockIdx.x] + 1;
    }
    for (uint64_t i = starterI, resultIx = starter; i < threadsInBlockNumberOfChecks; i++, resultIx++) {
        if (buffResult[resultIx]) {
            buffCombinedResult[blockIdx.x] = resultIx;
            buffResult[resultIx] = false;
            return;
        }
    }
    buffCombinedResult[blockIdx.x] = 0xffffffffffff;
}

__global__ void kernelUncompressed(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks, const uint32_t checksum) {
    uint64_t _start[5];
    beu32 d_hash[8];

    int64_t resIx = threadIdx.x;
    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_start, _stride, tIx);
    _add(_start, buffRangeStart);
    bool wasResult = false;
    initShared();
    for (uint64_t i = 0, resultIx = tIx; i < threadNumberOfChecks; i++, resultIx++) {
        if (_checksumDoubleSha256CheckUncompressed(checksum, d_hash, _start)) {
            _blockResults[resIx] = resultIx;
            if (!wasResult) {
                _blockResultFlag[0] = true;
            }
            wasResult = true;
            resIx += blockDim.x;
        }
        _add(_start, _stride);
    }
    summaryShared(gpuIx, unifiedResult, isResultFlag);
}
__global__ void kernelUncompressed(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks) {
    uint64_t _start[5];
    beu32 d_hash[8];

    int64_t resIx = threadIdx.x;
    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_start, _stride, tIx);
    _add(_start, buffRangeStart);
    bool wasResult = false;
    initShared();
    for (uint64_t i = 0, resultIx = tIx; i < threadNumberOfChecks; i++, resultIx++) {
        if (_checksumDoubleSha256CheckUncompressed(_start[0] & 0xffffffff, d_hash, _start)) {
            _blockResults[resIx] = resultIx;
            if (!wasResult) {
                _blockResultFlag[0] = true;
            }
            wasResult = true;
            resIx += blockDim.x;
        }
        _add(_start, _stride);
    }
    summaryShared(gpuIx, unifiedResult, isResultFlag);
}
__global__ void kernelCompressed(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks) {
    uint64_t _start[5];
    beu32 d_hash[8];
    int64_t resIx = threadIdx.x;
    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_start, _stride, tIx);
    _add(_start, buffRangeStart);
    bool wasResult = false;
    initShared();
    for (uint64_t i = 0, resultIx = tIx; i < threadNumberOfChecks; i++, resultIx++) {
        if (((_start[0] & 0xff00000000) >> 32) != 0x01) {
            _add(_start, _stride);
            continue;
        }
        if (_checksumDoubleSha256CheckCompressed(_start[0] & 0xffffffff, d_hash, _start)) {
            _blockResults[resIx] = resultIx;
            if (!wasResult) {
                _blockResultFlag[0] = true;
                wasResult = true;
            }            
            resIx += blockDim.x;            
        }
        _add(_start, _stride);
    }
    summaryShared(gpuIx, unifiedResult, isResultFlag);
}
__global__ void kernelCompressed(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag, uint64_t* const __restrict__ buffRangeStart, const int threadNumberOfChecks, const uint32_t checksum) {
    uint64_t _start[5];
    beu32 d_hash[8];

    int64_t resIx = threadIdx.x;
    int64_t tIx = (threadIdx.x + blockIdx.x * blockDim.x) * threadNumberOfChecks;
    IMult(_start, _stride, tIx);
    _add(_start, buffRangeStart);
    bool wasResult = false;
    initShared();
    for (uint64_t i = 0, resultIx = tIx; i < threadNumberOfChecks; i++, resultIx++) {
        if (((_start[0] & 0xff00000000) >> 32) != 0x01) {
            _add(_start, _stride);
            continue;
        }
        if (_checksumDoubleSha256CheckCompressed(checksum, d_hash, _start)) {
            _blockResults[resIx] = resultIx;
            if (!wasResult) {
                _blockResultFlag[0] = true;
                wasResult = true;
            }
            resIx += blockDim.x;
        }
        _add(_start, _stride);
    }
    summaryShared(gpuIx, unifiedResult, isResultFlag);
}

__device__ __inline__ void initShared() {
    for (int i = threadIdx.x; i < blockDim.x * 4;) {
        _blockResults[i] = UINT32_MAX;
        i += blockDim.x;
    }
    if (threadIdx.x == 0) {
        _blockResultFlag[0] = false;  
    }
    __syncthreads();
}
__device__ __inline__ void summaryShared(const int gpuIx, uint32_t* unifiedResult, bool* isResultFlag) {
    __syncthreads();
    if (threadIdx.x == 0 && _blockResultFlag[0]) {
        isResultFlag[gpuIx] = true;
        for (int i = 0, rIx = (blockIdx.x + 4*gpuIx*gridDim.x*blockDim.x); i < blockDim.x * 4; i++) {
            if (_blockResults[i] != UINT32_MAX) {
                unifiedResult[rIx] = _blockResults[i];
                rIx += gridDim.x;
            }
        }
    }
}

__device__  __inline__ bool _checksumDoubleSha256CheckCompressed(unsigned int checksum, beu32* d_hash, uint64_t* _start) {
    sha256Kernel(d_hash,
        _start[4] >> 16,
        (_start[4] & 0x0000ffff) << 16 | _start[3] >> 48,
        (_start[3] & 0xffffffffffff) >> 16,
        (_start[3] & 0x0000ffff) << 16 | _start[2] >> 48,
        (_start[2] & 0xffffffffffff) >> 16,
        (_start[2] & 0x0000ffff) << 16 | _start[1] >> 48,
        (_start[1] & 0xffffffffffff) >> 16,
        (_start[1] & 0x0000ffff) << 16 | _start[0] >> 48,
        ((_start[0] & 0xffffffffffff) >> 16) & 0xffff0000 | 0x8000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x110);

    return _checksumDoubleSha256(checksum, d_hash);
}

__device__  __inline__ bool _checksumDoubleSha256CheckUncompressed(unsigned int checksum, beu32* d_hash, uint64_t* _start) {
    sha256Kernel(d_hash,
        _start[4] >> 8,
        (_start[4] & 0x000000ff) << 24 | _start[3] >> 40,
        (_start[3] & 0xffffffffff) >> 8,
        (_start[3] & 0xff) << 24 | _start[2] >> 40,
        (_start[2] & 0xffffffffff) >> 8,
        (_start[2] & 0xff) << 24 | _start[1] >> 40,
        (_start[1] & 0xffffffffff) >> 8,
        (_start[1] & 0xff) << 24 | _start[0] >> 40,
        ((_start[0] & 0xffffffffff) >> 8) & 0xff000000 | 0x800000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x00000000,
        0x108);

    return _checksumDoubleSha256(checksum, d_hash);
}

__device__  __inline__ bool _checksumDoubleSha256(unsigned int checksum, beu32* d_hash) {
    sha256Kernel(d_hash, d_hash[0], d_hash[1], d_hash[2], d_hash[3], d_hash[4], d_hash[5],
        d_hash[6], d_hash[7], 0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000, 0x00000000, 0x100);
    return (checksum == d_hash[0]);
}

__device__  __inline__ void sha256Kernel(beu32* const hash, C16(COMMA, EMPTY)) {
#undef  H
#define H(i,alpha,magic)  beu32 hout##i;

    H8(EMPTY, EMPTY);

#undef  C
#define C(i)              c##i

#undef  H
#define H(i,alpha,magic)  &hout##i

    sha256_chunk0(C16(COMMA, EMPTY), H8(COMMA, EMPTY));

    //
    // SAVE H'S FOR NOW JUST SO NVCC DOESN'T OPTIMIZE EVERYTHING AWAY
    //
#undef  H
#define H(i,alpha,magic)  hash[i] = hout##i;

    H8(EMPTY, EMPTY);
}

__device__  __inline__ void _add(uint64_t* C, uint64_t* A) {
    __Add1(C, A);
}

__device__  __inline__ void _load(uint64_t* C, uint64_t* A) {
    __Load(C, A);
}


__device__  __inline__ void IMult(uint64_t* r, uint64_t* a, int64_t b) {
    uint64_t t[NBBLOCK];
    // Make b positive
    int64_t msk = b >> 63;
    int64_t nmsk = ~msk;
    b = ((-b) & msk) | (b & ~msk);
    USUBO(t[0], a[0] & nmsk, a[0] & msk);
    USUBC(t[1], a[1] & nmsk, a[1] & msk);
    USUBC(t[2], a[2] & nmsk, a[2] & msk);
    USUBC(t[3], a[3] & nmsk, a[3] & msk);
    USUB(t[4], a[4] & nmsk, a[4] & msk);
    Mult2(r, t, b)
}

cudaError_t loadStride(uint64_t* stride){
    return cudaMemcpyToSymbol(_stride, stride, 5 * sizeof(uint64_t));
}