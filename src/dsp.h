/*

    dsp.h

    This file is part of Clap Recognition Using a Neural Network from Scratch (C++ for Arduino): https://github.com/BojanJurca/Neural-Network-from-Scratch-for-Arduino-A-Clap-Recognition-System
    
    Digital signal processing of audio recordings - extracting features from sounds.
    
    Bojan Jurca, Nov 26, 2025

*/



// platform abstraction 
#ifdef ARDUINO                  // Arduino build requires LightwaightSTL library: https://github.com/BojanJurca/Lightweight-Standard-Template-Library-STL-for-Arduino
#else                           // standard C++ build
    #include <cstddef>
#endif


#ifndef __DSP_H__
    #define __DSP_H__

    #include "fft.h"
    #include "mel.h"
    #include "dct.h"
    
    array<float, featureCount> extractFeatures (const int (&soundRecording) [sampleCount]) {
        array<float, featureCount> features;

        #if ZCR == 1
            // feature [0] = zero crossings
            features [ZCR - 1] = 0;
            for (int s = 1; s < sampleCount; s++)
                    if (signbit (soundRecording [s - 1]) != signbit (soundRecording [s])) 
                            features [ZCR - 1] ++;

            // normalize this features = bring all the values somewhere in the range [0, 1]
            features [ZCR - 1] /= 30;                            
        #endif

        #if LRC == 1
            // feature [1] = linear regression coeficient
            int n = 0;
            float sumX = 0;
            float sumY = 0;
            float sumXY = 0;
            float sumX2 = 0;                
            for (int i = 0; i < sampleCount; i++) {                
                    float y = log10 (1.0 + (float) soundRecording [i] * (float) soundRecording [i]); // abs (soundRecording [i]);
                    float x = i;
                    n ++;
                    sumX += x;
                 sumY += y;
                    sumXY += x * y;
                    sumX2 += x * x;
            }
            features [ZCR + LRC - 1] = (sumXY * n - sumX * sumY) / (sumX2 * n - sumX * sumX);

            // normalize this features = bring all the values somewhere in the range [0, 1]
            features [ZCR + LRC - 1] = 0.002 - 100 * features [ZCR + LRC - 1];
        #endif

        // the rest of the features are mel filter values, but first we need the magnitudes from FFT
        float magnitude [distinctFftCoeficients]; // output from FFT
        {
            complex<float> fftInput [sampleCount];
            complex<float> fftOutput [sampleCount];
            for (int i = 0; i < sampleCount; i++)
                fftInput [i] = { (float) soundRecording [i], 0.f };
            fft (fftOutput, fftInput);
            for (int i = 0; i < distinctFftCoeficients; i++)
                magnitude [i] = abs (fftOutput [i]) / sampleCount;
        }

        // calculate MEL filters (&feature [ZCR + LRC], magnitude);
        float melFilters [melFilterCount];
        calculateMelFilters (melFilters, magnitude);

        float mfcc [melFilterCount]; // note that dct is performed on melFilterCount although we'll only use the first mfccCount coeficients
        dct (mfcc, melFilters);

        for (int i = 0; i < mfccCount; i++)
            features [ZCR + LRC + i] = mfcc [i];

        // normalize only the first coeficient, the other coefcients are already small enough
        features [ZCR + LRC] /= 30;

        return features;
    }

#endif