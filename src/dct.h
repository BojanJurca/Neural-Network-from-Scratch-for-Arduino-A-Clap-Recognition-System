/*

    dct.h

    This file is part of Clap Recognition Using a Neural Network from Scratch (C++ for Arduino): https://

    The discrete cosine transform (type II with scaling) computes the DCT according to its definition in O(n²) time, 
    as fast algorithms are often not applicable to arbitrary input data lengths.

    Bojan Jurca, Oct 10, 2025

*/


// platform abstraction 
#ifdef ARDUINO                  // Arduino build requires LightwaightSTL library: https://github.com/BojanJurca/Lightweight-Standard-Template-Library-STL-for-Arduino
#else                           // standard C++ build
    #include <cstddef>
#endif


#ifndef __DCT_H__
    #define __DCT_H__

    // Discrete cosine transform (DCT-II with scaling) - by definition - O (n²)
    template<typename T, size_t N>
    void dct (T (&output) [N], T (&input) [N]) {

        for (int k = 0; k < N; k ++) {
            double sum = 0.0;
            for (int n = 0; n < N; n ++)
                sum += input [n] * cos (M_PI * k * (2 * n + 1) / (2.0 * N));
            
           output [k] = sum * sqrt (2.0 / N);
            if (k == 0)
                output [k] *= 1.0 / sqrt (2); // scaling
        }
    
    }


    // Fast cosine transform - O (n log n)
    // gives the same result as DCT-II with scaling but only when N is a power of 2
    
    template <size_t N>
    struct IsPowerOf2 {
        static constexpr bool value = (N & (N - 1)) == 0;
    };
    
    template <size_t N, bool = IsPowerOf2<N>::value>
    struct NextPowerOf2Helper;
    
    template <size_t N>
    struct NextPowerOf2Helper<N, true> {
        static constexpr size_t value = N;
    };
    
    template <size_t N>
    struct NextPowerOf2Helper<N, false> {
        static constexpr size_t value = NextPowerOf2Helper<N + 1>::value;
    };



//void FastDctLee_transform(double vec[], size_t len) {

    template<typename T>
    void __forwardTransform__ (T vec [], T temp [], size_t N);


    template<typename T, size_t N>
    void fct (T (&output) [N], const T (&input) [N]) {
        static constexpr unsigned int M = NextPowerOf2Helper<N>::value;

        // copy the first N values to result
        T result [M];
        for (int i = 0; i < N; i++)
            result [i] = input [i];
        // mirror the rest of the values - this will give the coeficients closer to DCT II by definition when N is not a power of 2
        for (int i = 0; i < M - N; i++)
            if (N % 2 == 0) // N is evens number
                result [N + i] = input [N - 1 - i];
            else // N is odd number
                result [N + i] = input [N - 2 - i];

        // call (unscaled) __forwardTransform__ to calclulate DCT-II
	    T temp [M];
	    __forwardTransform__ (result, temp, M);

        // copy and scale the result to output
	    output [0] = result [0] * sqrt (1.0 / N);
        for (int i = 1; i < N; i++)
            output [i] = result [i] * sqrt (2.0 / N);
    }


    // fast DCT-II without scaling (Lee, according to https://github.com/nayuki/Nayuki-web-published-code/blob/master/fast-discrete-cosine-transform-algorithms/FastDctLee.cpp)
    template<typename T>
    void __forwardTransform__ (T vec [], T temp [], size_t N) {
    	if (N == 1)
    		return;
    	size_t halfN = N / 2;
    	for (size_t i = 0; i < halfN; i++) {
    		T x = vec [i];
    		T y = vec [N - 1 - i];
    		temp [i] = x + y;
    		temp [i + halfN] = (x - y) / (cos ((i + 0.5) * M_PI / N) * 2);
    	}

    	__forwardTransform__ (temp, vec, halfN);
    	__forwardTransform__ (&temp [halfN], vec, halfN);
    	
    	for (size_t i = 0; i < halfN - 1; i++) {
    		vec [i * 2 + 0] = temp [i];
    		vec [i * 2 + 1] = temp [i + halfN] + temp [i + halfN + 1];
    	}
    	vec [N - 2] = temp [halfN - 1];
    	vec [N - 1] = temp [N - 1];
    }

#endif
