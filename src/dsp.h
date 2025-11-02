/*

    dsp.h

    This file is part of Clap Recognition Using a Neural Network from Scratch (C++ for Arduino): https://github.com/BojanJurca/Neural-Network-from-Scratch-for-Arduino-A-Clap-Recognition-System
    
    Digital signal processing of audio recordings - extracting features from sounds.
    
    Bojan Jurca, Nov 26, 2025

*/


#ifndef __DSP_H__
    #define __DSP_H__

    #include "fft.h"
    #include "mel.h"
    #include "dct.h"
    
    
    void extractFeaturesFromSoundRecording (float feature [featureCount], int soundRecording [sampleCount]) {
    
        #if ZCR == 1
            // feature [0] = zero crossings
            feature [0] = 0;
            for (int s = 1; s < sampleCount; s++)
                    if (signbit (soundRecording [s - 1]) != signbit (soundRecording [s])) 
                            feature [0] ++;

            // normalize this features = bring all the values somewhere in the range [0, 1]
            feature [0] /= 30;                            
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
            feature [ZCR] = (sumXY * n - sumX * sumY) / (sumX2 * n - sumX * sumX);
            
            // normalize this features = bring all the values somewhere in the range [0, 1]
            feature [ZCR] = 0.002 - 100 * feature [ZCR];
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
            feature [ZCR + LRC + i] = mfcc [i]; // skip the first coeficient

        // normalize only the first coeficient, the other coefcients are already small enough
        feature [ZCR + LRC] /= 30;
    }

#endif