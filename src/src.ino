/*

    Arduino_neuralNetwork.ino

    This file is part of Clap Recognition Using a Neural Network from Scratch (C++ for Arduino): https://github.com/BojanJurca/Neural-Network-from-Scratch-for-Arduino-A-Clap-Recognition-System

    Sound sampling and clap recognition using neural network on Arduino Mega 2560

    Bojan Jurca, Nov 26, 2025

*/


// input: sampling definitions

    #define microphonePin 0             // the A/D pin you microphone is connected to the controller
    #define signalMid 511               // obtained by measuring the silence mean value of a particular microphone
    #define signalTreshold 350          // the treshold when the sound gets loud enough to indicate it may be a clap
    #define signalOverloaded 509        // obtained by observing the microphone signal - this is when the signal gets overloaded
    #define sampleCount 256             // 256 samples sampled with 35.75 kHz gives 7.16 ms of the sound
    #define samplingFrequency 35750.0   // sampling frequency is determined by setting A/D prescale bits
    
// output: LED

    #define ledPin 2                    // the pin your LED is connected
    
// discrete Fourier transform

    #define NyquistFrequency ( samplingFrequency / 2 )
    #define distinctFftCoeficients  ((sampleCount + 1) / 2 + 1) // works for both, even and odd numbers

// mel filters and MFCCs

    #define melFilterCount 20
    #define mfccCount 13        // must be < melFilterCount, tipically 12 or 13

// sound pattern features

    #define ZCR 1 // 1 - use this feature, 0 - don't use this feature
    #define LRC 1 // 1 - use this feature, 0 - don't use this feature

    #define featureCount ( ZCR + LRC + mfccCount ) // feature extracted from sound recordings as the input to the neural network
    // feature [0]              <- zerro crossings - the number the signal crosses time axes
    // feature [1]              <- linear regression coeficient - how fast the signal amplitude is dropping    
    // feature [2, ...]         <- MFCC [1], ...

// include a small compatibility.h library with which the same code will run on Arduino Mega 2560 and desktop computer as well

    #include "compatibility.h" // Arduino <-> standard C++ compatibility

// void extractFeaturesFromSoundRecording (float feature [featureCount], int soundRecording [sampleCount]);

    #include "dsp.h"


// ----- the neural network -----
 

    #include "neuralNetwork.hpp"
    
    //                    .--- the number of neurons in the first layer - the number of input sound features in our case
    //                    |                .--- second layer activation function
    //                    |                |    .--- the number of neurons in the second layer
    //                    |                |    |      .--- third layer activation function
    //                    |                |    |      |    .--- the number of neurons in the third layer
    //                    |                |    |      |    |                                     .--- output layer activation function
    //                    |                |    |      |    |                                     |    .--- the number of neurons in the output layer - it corresponds to the number of categories that the neural network recognizes (clap or not a clap in our case)
    //                    |                |    |      |    |                                     |    |
    neuralNetworkLayer_t<featureCount, Sigmoid, 2, Sigmoid, 3,/* add more layers if needed */ Sigmoid, 2> neuralNetwork; // this configuration gives good results but try others as well  
    // at this point neuralNetwork is initialized with random weights and biases and it is ready for training
    // - you can either start training it and export the trained model when fiished
    // - or you can load already trained model that is cappable of making usable outputs 


// ----- the input -----

    // non-blocking reading of analog microphone with 35.75 kHz sampling frequency, see: https://www.gammon.com.au/adc
    bool ADCinProgress;
    void analogReadBegin () { 
            bitSet (ADCSRA, ADSC); // start a conversion
            ADCinProgress = true;
    }
    int analogReadEnd () {
            while (!bit_is_clear (ADCSRA, ADSC));
            ADCinProgress = false;
            return ADC; // read the result
    }
    
    // queue buffer to capture the signal
    template<typename T>
        class soundQueue_t {
                T __buf__ [256] = {};
                unsigned char __p__ = 0;
                int __overloadCount__ = 0;
        
            public:
                void push (int e) {
                        if (abs (__buf__ [__p__]) >= signalOverloaded)
                                __overloadCount__ --;
                        if (abs (e) >= signalOverloaded)
                                __overloadCount__ ++;
                    __buf__ [__p__] = e;
                    __p__ += 1; // __p__ = (__p__ + 1) % 256;
                }
                inline T &front () __attribute__((always_inline)) { return __buf__ [__p__]; }
                inline int overloaded () __attribute__((always_inline)) { return __overloadCount__; }
                inline T &operator [] (unsigned char position) __attribute__((always_inline)) { return __buf__ [(unsigned char) (__p__ + position)]; } // (__p__ + position) % 256
                inline size_t size () __attribute__((always_inline)) { return 256; }
                void clear () { 
                        memset ((void *) __buf__, 0, sizeof (__buf__)); 
                        __overloadCount__ = 0;
                }
        };
        
        soundQueue_t<int> rawSamples;


// ----- setup, training the neural network or just loading already trained model, input and output initialization -----

    void setup () {
        cinit (); // instead of Serial.begin (9600); or Serial.begin (115200);
        
        // at this point neuralNetwork is initialized with random weights and biases and it is ready for training
        // - you can either start training it and export the trained model when fiished
        // - or you can load already trained model that is cappable of making usable outputs 
        

        // load trained model 

        /**/
            // ----- LOAD TRAINED MODEL BEGIN -----

            cout << "loading trained model\n";
            // error over all patterns: 3.47867
            // accuracy: 100% (claps: 100%   others: 100%)
            // model:
            const int32_t model [] PROGMEM = {  -1065966516,-1079298553,1082273279,1073625518,1061579350,1077900222,1070075945,-1069910552,-1072431041,1075562182,
                                                1067040550,-1088575706,1074520146,-1068238440,1083190075,1077271283,1063469639,1069772027,-1072093467,1051662872,
                                                1072974707,1082777248,1070508301,-1081046202,-1073672553,-1077037630,-1086993185,-1089753020,-1065866667,-1086648393,
                                                1074673962,1057196553,1086869673,1083529437,-1060619410,-1063640277,1053098129,1041190271,-1069148155,1078435292,
                                                -1097106015,-1052773919,1095404895,-1084430925,1094412215,-1051756565,1059624795,-1121848440,1054633459};
            neuralNetwork = model;
                       
            // ----- LOAD TRAINED MODEL END -----    
        /**/
        
        /*
            // ----- TRAINING BEGIN -----
            cout << "----- TRAINING BEGIN -----\n";
            
            // train the neural network with training patterns, this easier done on bigger computers, it doesn't have to be an Arduino
            #include "trainingRecordings.h"
            constexpr int eachSetCount = min (clapRecordingCount, otherRecordingCount);

            cout << "   " << eachSetCount << " of clap feature patterns\n";
            cout << "   " << eachSetCount << " of other sound feature patterns\n";
            cout << "   80 % of them will be used for training and 20 % for validation\n";
            
            // do trainingCount independent trainings, most likely each ending in a different local minimum
            #define trainingCount 100 // wel'll do trainingCount of independed trainings, most likely each time ending in a different local minimum

            // store the best training result
            float bestTrainingErrorOverAllPatterns = 1.0f / 0.0f;
            float bestTrainingAccuracy = 0;
            float bestTrainingClapAccuracy = 0;
            float bestTrainingOtherAccuracy = 0;
            byte bestTrainingModel [sizeof (neuralNetwork)];
        
            for (int t = 0; t < trainingCount; t++) {
                cout << "\n   " << t << ". training\n";

                // extract features from clap audio_recordings
                float clapFeatures [clapRecordingCount][featureCount];
                for (int i = 0; i < clapRecordingCount; i++) {
                    extractFeaturesFromSoundRecording (clapFeatures [i], clapRecording [i]);
                    // cout << "   {";
                    // for (size_t f = 0; f < featureCount; f++) {
                    //         if (f)
                    //                 cout << ",";
                    //         char buf [25];
                    //         #ifdef ARDUINO_ARCH_AVR // Assuming Arduino Mega or Uno
                    //                 dtostrf (clapFeatures [i][f], 1, 4, buf);
                    //         #else
                    //                 snprintf (buf, sizeof (buf), "%.4f", clapFeatures [i][f]);
                    //         #endif
                    //         cout << buf << "f";
                    // }
                    // cout << "},\n";
                }

                // extract features from other audio_recordings
                float otherFeatures [otherRecordingCount][featureCount];
                for (int i = 0; i < otherRecordingCount; i++) {
                    extractFeaturesFromSoundRecording (otherFeatures [i], otherRecording [i]);
                    // cout << "   {";
                    // for (size_t f = 0; f < featureCount; f++) {
                    //         if (f)
                    //                 cout << ",";
                    //         char buf [25];
                    //         #ifdef ARDUINO_ARCH_AVR // Assuming Arduino Mega or Uno
                    //                 dtostrf (otherFeatures [i][f], 1, 4, buf);
                    //         #else
                    //                 snprintf (buf, sizeof (buf), "%.4f", otherFeatures [i][f]);
                    //         #endif
                    //         cout << buf << "f";
                    // }
                    // cout << "},\n";                
                }

                // store the best epoch result
                float bestEpochErrorOverAllPatterns = 1.0f / 0.0f;
                float bestEpochAccuracy = 0;
                float bestEpochClapAccuracy = 0;
                float bestEpochOtherAccuracy = 0;

                #define epoch 5000
                int eCount = epoch;
                int e;
                for (e = 0; e < eCount; e++) {
                    float errorOverAllPatterns = 0;
                     
                    // training   
                    for (size_t i = 0; i < eachSetCount; i++) {
                        if (i % 5 != 0) { // leave every 5th pattern for validation
                            errorOverAllPatterns += neuralNetwork.backwardPropagation (clapFeatures [i], {1, 0});   // index 0 = clap
                            errorOverAllPatterns += neuralNetwork.backwardPropagation (otherFeatures [i], {0, 1});  // index 1 = other sound
                        }
                    }
                    
                    // validation
                    int clapsValidated = 0;
                    int clapsRecognized = 0;
                    for (int i = 0; i < eachSetCount; i++) {
                        if (i % 5 == 0) { // take every 5th pattern for validation
                            clapsValidated ++;
                            auto probability = neuralNetwork.forwardPass (clapFeatures [i]);
                            if (probability [0] > 0.5) // index 0 = clap
                                clapsRecognized ++;
                        }
                    }
                    int othersValidated = 0;
                    int othersRecognized = 0;
                    for (size_t i = 0; i < eachSetCount; i++) {
                        if (i % 5 == 0) { // take every 5th pattern for validation
                            othersValidated ++;
                            auto probability = neuralNetwork.forwardPass (otherFeatures [i]);
                            if (probability [1] >= 0.5) // index 1 = other sound
                                othersRecognized ++;
                        }
                    }
                                        
                    float accuracy = (float) (clapsRecognized +  othersRecognized) / (float) (clapsValidated + othersValidated);
                        if (accuracy >= bestEpochAccuracy) {
                            bestEpochAccuracy = accuracy;
                            bestEpochClapAccuracy = (float) clapsRecognized / (float) clapsValidated;
                            bestEpochOtherAccuracy = (float) othersRecognized / (float) othersValidated;
                            
                            cout << "      " << e << "   error over all patterns: " << errorOverAllPatterns << "        \r";

                            if (bestEpochAccuracy > bestTrainingAccuracy ||(bestEpochAccuracy == bestTrainingAccuracy && errorOverAllPatterns < bestTrainingErrorOverAllPatterns)) {
                                bestTrainingErrorOverAllPatterns = errorOverAllPatterns;
                                bestTrainingAccuracy = bestEpochAccuracy;
                                bestTrainingClapAccuracy = bestEpochClapAccuracy;
                                bestTrainingOtherAccuracy = bestEpochOtherAccuracy;
                                memcpy (&bestTrainingModel, &neuralNetwork, sizeof (neuralNetwork));
                            }
                                
                            eCount = epoch;
                        } else {
                            // already overtrained? 
                            if (eCount == epoch && bestEpochAccuracy > 0.9 && e >= 900)
                                eCount = e + 100; // try another 100 iterations if we can get out of this
                        }
                }
                cout << "\n         training stopped at " << e << " iterations, accuracy: " << bestEpochAccuracy * 100 << "% (claps: " << bestEpochClapAccuracy * 100 << "%   others: " << bestEpochOtherAccuracy * 100 << "%)\n";        

                
                // start the next training with different random seed
                srand (static_cast<unsigned> (time (nullptr)));
                neuralNetwork.randomize ();
            }
        
            memcpy (&neuralNetwork, &bestTrainingModel, sizeof (neuralNetwork));
            cout << "\n   --------------------------------------\n";
            cout << "   error over all patterns: " << bestTrainingErrorOverAllPatterns << endl;
            cout << "   accuracy: " << bestTrainingAccuracy * 100 << "% (claps: " << bestTrainingClapAccuracy * 100 << "%   others: " << bestTrainingOtherAccuracy * 100 << "%)\n";        
            cout << "   model:\n";
            cout << neuralNetwork << endl;

            // ----- TRAINING END -----
            cout << "----- TRAINING END -----\n";
        */
        

        cout << "\nclap recognition system is ready\n\n";

        // initialize the input: increase analogReading speed by changing prescale bits and make AD conversion non blocking (https://www.gammon.com.au/adc)
        ADCSRA =  bit (ADEN);   // turn ADC on

        ADCSRA &= ~(bit (ADPS0) | bit (ADPS1) | bit (ADPS2)); // clear prescaler bits
        // uncomment as required
        // ADCSRA |= bit (ADPS0);                               // prescaler   2 -> sampling frequency = 200.20 kHz 
        // ADCSRA |= bit (ADPS1);                               // prescaler   4 -> sampling frequency = 200.20 kHz 
        // ADCSRA |= bit (ADPS0) | bit (ADPS1);                 // prescaler   8 -> sampling frequency = 125.12 kHz   
        // ADCSRA |= bit (ADPS2);                               // prescaler  16 -> sampling frequency = 66.73 kHz  
        ADCSRA |= bit (ADPS0) | bit (ADPS2);                 // prescaler  32 -> sampling frequency = 35.75 kHz 
        // ADCSRA |= bit (ADPS1) | bit (ADPS2);                 // prescaler  64 -> sampling frequency = 17.87 kHz  
        // ADCSRA |= bit (ADPS0) | bit (ADPS1) | bit (ADPS2);   // prescaler 128 -> sampling frequency = 8.94 kHz 

        ADMUX = bit (REFS0) | (microphonePin & 0x07);  // AVcc           

        // start the first AD conversion
        analogReadBegin ();
        
        // initialize the output
        pinMode (ledPin, INPUT | OUTPUT);
        digitalWrite (ledPin, 0);
    }



// ----- loop -----

    void loop () {
   
        // measure signalMid in silence
        /*
                static unsigned long cnt = 0;
                static unsigned long long sum = 0;

                sum += analogReadEnd ();
                analogReadBegin ();
                if (++cnt == 10000) {
                        Serial.print ("signalMid = "); Serial.println ( (float) sum / (float) cnt );
                        cnt = 0;
                        sum = 0;
                }
                return;
        */

        rawSamples.push ( analogReadEnd () - signalMid ); // wait for previous conversion to finish and return the result
        analogReadBegin (); // start the next AD conversion immediatelly

        // measure sampling frequency
        /* 
                static unsigned long startMillis = millis ();
                static unsigned int cnt = 0;
                if (cnt++ == 1000) {
                        unsigned long endMillis = millis ();        
                        cout << "sampling frequency = " << (float) cnt / (float) (endMillis - startMillis) << " kHz" << endl;
                        while (true);
                }
                return;
        */

        if (abs (rawSamples.front ()) >= signalTreshold && rawSamples.overloaded () == 0) {

            int soundRecording [sampleCount];
            for (int i = 0; i < sampleCount; i++)
                soundRecording [i] = rawSamples [i];
            
            float feature [featureCount];
            extractFeaturesFromSoundRecording (feature, soundRecording);
                /*
            for (int i = 0; i < featureCount; i++)
                cout << feature [i] << endl;
                */


            // ask neural network what it thinks about these features
            float p = neuralNetwork.forwardPass (feature) [0];
            if (p > 0.5) {
                    cout << "    it's a clap, p = " << p << endl;
                    digitalWrite (ledPin, !digitalRead (ledPin));
            } else {
                    cout << "        not a clap, p = " << p << endl;
            }


            delay (100); // wait before reading the next clap
            rawSamples.clear ();
            analogReadEnd ();
            analogReadBegin ();
        }

    }