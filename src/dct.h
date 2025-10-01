/*

    dct.h

    This file is part of Clap Recognition Using a Neural Network from Scratch (C++ for Arduino): https://github.com/BojanJurca/Neural-Network-from-Scratch-for-Arduino-A-Clap-Recognition-System

    The discrete cosine transform (type II with scaling) computes the DCT according to its definition in O(n²) time, 
    as fast algorithms are often not applicable to arbitrary input data lengths.

    Bojan Jurca, Oct 10, 2025

*/


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

#endif
