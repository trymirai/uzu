#ifndef rng_metal
#define rng_metal

#include <metal_stdlib>
using namespace metal;

/* ───────────── Philox4x32-10 RNG ───────────── */

#define PHILOX_W32_0   0x9E3779B9u
#define PHILOX_W32_1   0xBB67AE85u
#define PHILOX_M4x32_0 0xD2511F53u
#define PHILOX_M4x32_1 0xCD9E8D57u

struct PhiloxState {
    uint key[2];
    uint ctr[4];
    int state_idx;
    uint output[4];
};

inline uint mulhilo32(uint a, uint b, thread uint* hip) {
    uint64_t product = (uint64_t(a)) * (uint64_t(b));
    *hip = uint(product >> 32);
    return uint(product);
}

inline void philox4x32round(thread uint* ctr, thread uint* key) {
    uint hi0, hi1;
    uint lo0 = mulhilo32(PHILOX_M4x32_0, ctr[0], &hi0);
    uint lo1 = mulhilo32(PHILOX_M4x32_1, ctr[2], &hi1);

    uint new_ctr[4];
    new_ctr[0] = hi1 ^ ctr[1] ^ key[0];
    new_ctr[1] = lo1;
    new_ctr[2] = hi0 ^ ctr[3] ^ key[1];
    new_ctr[3] = lo0;
    
    for(int i = 0; i < 4; i++) {
        ctr[i] = new_ctr[i];
    }
}

inline void philox4x32bumpkey(thread uint* key) {
    key[0] += PHILOX_W32_0;
    key[1] += PHILOX_W32_1;
}

inline void curand_philox4x32_10(thread uint* ctr, thread uint* key, thread uint* out) {
    uint local_ctr[4] = {ctr[0], ctr[1], ctr[2], ctr[3]};
    uint local_key[2] = {key[0], key[1]};
    
    // 10 rounds
    philox4x32round(local_ctr, local_key);                                   // 1
    philox4x32bumpkey(local_key); philox4x32round(local_ctr, local_key);     // 2
    philox4x32bumpkey(local_key); philox4x32round(local_ctr, local_key);     // 3
    philox4x32bumpkey(local_key); philox4x32round(local_ctr, local_key);     // 4
    philox4x32bumpkey(local_key); philox4x32round(local_ctr, local_key);     // 5
    philox4x32bumpkey(local_key); philox4x32round(local_ctr, local_key);     // 6
    philox4x32bumpkey(local_key); philox4x32round(local_ctr, local_key);     // 7
    philox4x32bumpkey(local_key); philox4x32round(local_ctr, local_key);     // 8
    philox4x32bumpkey(local_key); philox4x32round(local_ctr, local_key);     // 9
    philox4x32bumpkey(local_key); philox4x32round(local_ctr, local_key);     // 10
    
    for(int i = 0; i < 4; i++) {
        out[i] = local_ctr[i];
    }
}

inline void philox_state_incr(thread PhiloxState* s) {
    ++s->ctr[0];
    if(s->ctr[0] != 0) return;
    ++s->ctr[1];
    if(s->ctr[1] != 0) return;
    ++s->ctr[2];
    if(s->ctr[2] != 0) return;
    ++s->ctr[3];
}

inline void philox_init(thread PhiloxState* state, uint64_t seed, uint64_t offset) {
    // Initialize key from seed
    state->key[0] = uint(seed);
    state->key[1] = uint(seed >> 32);
    
    // Initialize counter from offset
    state->ctr[0] = uint(offset);
    state->ctr[1] = uint(offset >> 32);
    state->ctr[2] = 0;
    state->ctr[3] = 0;
    
    state->state_idx = 0;
    
    // Generate first batch of random numbers
    curand_philox4x32_10(state->ctr, state->key, state->output);
    philox_state_incr(state);
}

inline uint philox_next(thread PhiloxState* state) {
    if(state->state_idx >= 4) {
        curand_philox4x32_10(state->ctr, state->key, state->output);
        philox_state_incr(state);
        state->state_idx = 0;
    }
    return state->output[state->state_idx++];
}

inline float uniform_float(thread PhiloxState* state) {
    return float(philox_next(state)) * (1.0f/4294967296.0f);  /* (0,1) */
}

#endif /* rng_metal */
