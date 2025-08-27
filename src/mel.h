/*

    mel.h

    This file is part of Clap Recognition Using a Neural Network from Scratch (C++ for Arduino): https://github.com/BojanJurca/Neural-Network-from-Scratch-for-Arduino-A-Clap-Recognition-System

    Data tables to implement mel filters are calculated in advance to achieve better performace at run-time.
    
    Bojan Jurca, Sep 9, 2025

*/

#ifndef __MEL_H__
    #define __MEL_H__

 
    #define melPointCount  ( melFilterCount + 2 )
 
    #define maxMelFrequency ( 2595.0 * log10 (1.0 + NyquistFrequency / 700) )
    #define melFrequencyStep ( maxMelFrequency / (melPointCount - 1) )

    #define fftFrequencyStep ( samplingFrequency / sampleCount )
    

    // calculate the array of mel frequencies that correspond to FFT frequencies
    array<float, distinctFftCoeficients> generateMelFftFrequencies () {
        array<float, distinctFftCoeficients> result = {};
        for (size_t i = 0; i < distinctFftCoeficients; i ++)
            result [i] = static_cast<float>( 2595.0 * log10 (1 + (i * fftFrequencyStep) / 700) );
        return result;
    }
    auto melFftFrequency = generateMelFftFrequencies ();


    // calculate the array of mel points
    array<float, melPointCount> generateMelPoints () {
        array<float, melPointCount> result = {};
        for (size_t i = 0; i < melPointCount; i ++)
            result [i] = static_cast<float>( i * maxMelFrequency / (melPointCount - 1) );
        return result;
    }
    auto melPoint = generateMelPoints ();
    // mel filter 0 is defined by triangle melPoint [0], melPoint [1], melPoint [2]
    // mel filter 1 is defined by triangle melPoint [1], melPoint [2], melPoint [3]
    // ...


    void calculateMelFilters (float *melFilterValue, float magnitude [distinctFftCoeficients]) { // melFilterValue is array floatMelFIleterValue [melFilterCount]
        /*
        cout << "   mel points\n";
        for (int i = 0; i < melPointCount; i++)
            cout << "   " << melPoint [i] << "\n";
            
        cout << "   mel FFT frequencies\n";
        for (int i = 0; i < distinctFftCoeficients; i++)
            cout << "      " << melFftFrequency [i] << "\n";
        */

        // calculate the first range of FFT coeficients (for first half of mel triangle filter)
        int fromFftCoeficientIndex = 0;

        int toFftCoeficientIndex;
        for (int i = 0; i < distinctFftCoeficients; i++)
            if (melFftFrequency [i] < melPoint [1])
                toFftCoeficientIndex = i;
            else
                break;

        for (int f = 0; f < melFilterCount; f++) {

            //cout << "         fromFftCoeficientIndex = " << fromFftCoeficientIndex << "\n";
            //cout << "         toFftCoeficientIndex   = " << toFftCoeficientIndex << "\n";
            //cout << "         melFrequencyStep       = " << melFrequencyStep << "\n";

            // calculate the first slope of triange mel filter 
            melFilterValue [f] = 0;
            
            for (int i = fromFftCoeficientIndex; i <= toFftCoeficientIndex; i++)
                melFilterValue [f] += magnitude [i] * (melPoint [f + 1] - melFftFrequency [i]) / (melFrequencyStep);
                
            // calculate the second slope of triange mel filter and the second range of FFT coeficients (for second half of mel triangle filter)
            fromFftCoeficientIndex = toFftCoeficientIndex + 1;
            for (int i = fromFftCoeficientIndex; i < distinctFftCoeficients; i++)
                if (melFftFrequency [i] < melPoint [f + 2])
                    toFftCoeficientIndex = i;
                else
                    break;
                    
            for (int i = fromFftCoeficientIndex; i <= toFftCoeficientIndex; i++)
                melFilterValue [f] += magnitude [i] * (melPoint [f + 1] - melFftFrequency [i]) / (-melFrequencyStep);
                
            melFilterValue [f] = log10 (melFilterValue [f]); // apply logarithmic scale to (energy) magnitude
            
            // note that the first range of FFT coeficients for first half of the next mel triangle filter is exacltly the same as already calculated for second half of this mel triangle filter
        }
    }
 
#endif