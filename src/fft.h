/*

    fft.h

    This file is part of Clap Recognition Using a Neural Network from Scratch (C++ for Arduino): https://github.com/BojanJurca/Neural-Network-from-Scratch-for-Arduino-A-Clap-Recognition-System

    Fast Fouriere Transform using Radix-2 algorithm with complexity of O (n log n), according to https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm

    Bojan Jurca, May 22, 2025

*/


#ifndef __FFT_H__
    #define __FFT_H__


    template<typename T, size_t N>
    void fft (complex<T> (&output) [N], const complex<T> (&input) [N]) {
        static_assert((N > 0) && ((N & (N - 1)) == 0), "N must be a power of 2");

        // Calculate the number of stages to perform: 2^S = N
        int S = 0;
        for (int s = 1; s < N; s *= 2)
            S++;

        // Bit-reverse copy: output <- input
        for (size_t i = 0; i < N; i++) {
            size_t iTmp = i;
            size_t iReversed = 0;
            for (int s = 1; s <= S; s++) {
                iReversed = (iReversed << 1) | (iTmp & 0x01);
                iTmp >>= 1;
            }
            output [iReversed] = input [i];
        }

        // Go through all the stages
        for (int s = 1; s <= S; s++) {
            int m = 1 << s; // m = 2^s
            complex<T> omegam = exp (complex<T> (0, -2 * M_PI / m));
            for (size_t k = 0; k < N; k += m) {
                complex<T> omega = { 1.f, 0.f };
                for (size_t j = 0; j < m / 2; j++) {
                    complex<T> t = omega * output [k + j + m / 2];
                    complex<T> u = output [k + j];
                    output [k + j] = u + t;
                    output [k + j + m / 2] = u - t;
                    omega *= omegam;
                }
            }
        }
    }

#endif
