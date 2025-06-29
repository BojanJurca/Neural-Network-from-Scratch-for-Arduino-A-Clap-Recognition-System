/*

    Arduino_neuralNetwork.ino

    This file is part of Clap Recognition Using a Neural Network from Scratch (C++ for Arduino): https://github.com/BojanJurca/Clap-Recognition-Using-a-Neural-Network-from-Scratch-Cpp-for-Arduino

    Sound sampling and clap recognition using neural network on Arduino Mega 2560

    Bojan Jurca, May 22, 2025

*/



#include "compatibility.h" // Arduino <-> standard C++ compatibility

// ----- the input -----

    #define microphonePin 0
    
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
    
    #define soundMid 511        // obtained by measuring the silence mean value of a particular microphone
    #define soundTreshold 350   // the treshold when the sound gets loud enough to indicate it may be a clap
    #define soundOverloaded 509 // obtained by observing the microphone signal - this is when the signal gets overloaded
    #define rawSamplesCount 256 // 256 samples sampled with 35.75 kHz gives 7.16 ms of the sound

    // queue buffer to capture the signal
    template<typename T>
        class soundQueue_t {
                T __buf__ [256] = {};
                unsigned char __p__ = 0;
                int __overloadCount__ = 0;
        
            public:
                void push (int e) {
                        if (abs (__buf__ [__p__]) >= soundOverloaded)
                                __overloadCount__ --;
                        if (abs (e) >= soundOverloaded)
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


// ----- the output -----

    #include "fft.h"

    // we'll use two LEDs triggered by the clap - one by neural network, the other by rules - just for comparison of both approaches 
    #define ruleBasedLedPin 4
    #define neuralNetworkBasedLedPin 2


// ----- features extracted from clap sound as input to the neural network

    #define featureCount 8 // we'll use 8 feature extracted from clap sound as the input to the neural network
    // feature [0]              <- zerro crossings - the number the signal crosses time axes
    // feature [1]              <- linear regression coeficient - how fast the signal amplitude is dropping    
    // feature [2, 3]           <- energy - signal energy in te first and the second half of audio recording
    // feature [4, 5, 6, 7]     <- maginudes in 4 frequency bands


// ----- neural network -----
 

    #include "neuralNetwork.hpp"
    
    //                    .--- the number of neurons in the first layer - the number of input sound features in our case
    //                    |              .--- second layer activation function
    //                    |              |    .--- the number of neurons the second layer 
    //                    |              |    |                             .--- output layer activation function
    //                    |              |    |                             |      .--- the number of neurons in the output layer - it corresponds to the number of categories that the neural network recognizes (clap or not a clap in our case)
    //                    |              |    |                             |      |
    neuralNetworkLayer_t<featureCount, ReLU, 11, /* add more if needed */ Sigmoid, 2> neuralNetwork;

    
    // at this point neuralNetwork is initialized with random weights and biases and it is ready for training
    // - you can either start training it and export the trained model when fiished
    // - or you can load already trained model that is cappable of making usable outputs 


#include "claps.h"
#include "other_sounds.h"

// ----- setup, training the neural network or just loading already trained model -----

    void setup () {
        cinit (); // instead of Serial.begin (9600); or Serial.begin (115200);


        // load trained model 
        //    or 
        // train the neural network with training patterns

        /**/
                // import trained model

                cout << "\n----- Importing trained model -----\n\n";

                const int32_t model [] PROGMEM =        {-1102000500,-1094027232,1067359819,-1107615579,-1075348309,1067875840,1063310163,1049995543,1076519184,-1085001279,-1066981228,1075712476,-1073841140,-1095825720,1071055885,1065862261,
                                                        1058443171,-1096072972,1061262024,-1089628049,-1075247815,1075171395,-1090195372,-1095038395,-1088057270,1070012757,-1093271376,1074647097,1080688246,-1067216272,1068552050,1047563678,
                                                        1059499520,-1084466895,1065874458,-1075538078,-1067529226,1082844961,1041376978,-1089398971,1034207196,-1096448109,-1098658344,1035086773,1016397219,-1109503176,1049408262,-1125049158,
                                                        1068932400,-1079205083,1054946771,-1092715020,-1089633612,1068553930,1057583057,1050939432,1053390543,1049545560,1025518432,1058845790,-1106581644,1049875482,1019329447,1047160999,
                                                        1078159979,1034828790,-1066375831,1076963300,-1072682907,-1079190164,1068666699,1065940226,1053886155,1050510222,-1091070981,1051312640,1063652597,-1091281421,1027861246,1049449145,
                                                        -1091294220,1052287542,1039632578,-1123452964,1007740464,-1101371606,1032521949,-1085763048,-1091687296,-1093894772,-1091123669,1066696221,-1087693504,0,-1132126851,-1123627023,
                                                        -1095940337,1040433038,-1106748102,1068392508,-1065800384,1075926285,-1064933301,1081760402,-1095707286,1067178248,1042954168,-1064422411,-1082853035,1066316497,-1076948023,1082536966,
                                                        -1075062161,1082440538,-1064510678,1052674815,-1080046979,1033864988,1082549200,1057094739,-1096518692,-1097359117,1054632022};
                neuralNetwork = model;
        /**/
        
        /*
        
                // train the neural network - you can do this on bigger computer, doesn't have to be an Arduino


                cout << "\n----- Training -----\n\n";
        

                size_t clapCount = sizeof (clap) / (sizeof (float) * featureCount);
                // size_t snapCount = sizeof (snap) / (sizeof (float) * featureCount);
                size_t otherSoundCount = sizeof (otherSound) / (sizeof (float) * featureCount);

                cout << "\ntraining\n";            

                // This part, including testing different typologies, can be done more efficiently on larger computers and not necessarily on a controller,
                // as Arduino code is portable to standard C++.

                #define epoch 10000 // choose the right number of training iterations so the model gets trained but not overtrained
                for (int step = 0; step < epoch; step++) {
                        float errorOverAllPatterns = 0;
                        
                        for (size_t i = 0; i < clapCount; i++) 
                                if (i % 6 != 0) // leave every 5th pattern for testing
                                        errorOverAllPatterns += neuralNetwork.backwardPropagation (clap [i], {1, 0}); // index 0 = clap

                        for (size_t i = 0; i < otherSoundCount; i++) 
                                if (i % 6 != 0) // leave every 5th pattern for testing
                                        errorOverAllPatterns += neuralNetwork.backwardPropagation (otherSound [i], {0, 1}); // index 1 = another sound

                        cout << "   step " << step << " error over all patterns = " << errorOverAllPatterns << endl;
                }

                // test success
                
                cout << "\ntesting clap recognition\n";        
                for (size_t i = 0; i < clapCount; i++) 
                        if (i % 5 == 0) { // take every 5th pattern for testing
                            cout << "   clap [" << i << "]: ";
                            auto probability = neuralNetwork.forwardPass (clap [i]);
                            if (probability [0] > 0.5) // index 0 = clap
                                cout << "OK,    probabilities: ( ";
                            else
                                cout << "WRONG, probabilities: ( ";
                            for (auto p : probability)
                                cout << p << " ";
                            cout << ")\n";
                        }
                cout << "\ntesting other sounds\n";   
                for (size_t i = 0; i < otherSoundCount; i++) 
                        if (i % 5 == 0) { // take every 5th pattern for testing
                            cout << "   otherSound [" << i << "]: ";
                            auto probability = neuralNetwork.forwardPass (otherSound [i]);
                            if (probability [1] > 0.5) // index 0 = clap
                                cout << "   OK,    probabilities: ( ";
                            else
                                cout << "   WRONG, probabilities: ( ";
                            for (auto p : probability)
                                cout << p << " ";
                            cout << ")\n";                                    
                        }

                // export trained model
                cout << "\ntrained model: " << neuralNetwork << endl;
        */
        
     
        cout << "\n----- Clap recognition system is ready -----\n\n";


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
        pinMode (ruleBasedLedPin, INPUT | OUTPUT);
        digitalWrite (ruleBasedLedPin, 0);
        pinMode (neuralNetworkBasedLedPin, INPUT | OUTPUT);
        digitalWrite (neuralNetworkBasedLedPin, 0);        
    }



// ----- loop -----

    void loop () {
    
        // measure soundMid in silence
        /*
                static unsigned long cnt = 0;
                static unsigned long long sum = 0;

                sum += analogReadEnd ();
                analogReadBegin ();
                if (++cnt == 10000) {
                        Serial.print ("soundMid = "); Serial.println ( (float) sum / (float) cnt );
                        cnt = 0;
                        sum = 0;
                }
                return;
        */
        
        // see what microphone is producing with Arduino Serial Plotter
        /*
                cout << analogReadEnd () - soundMid << endl;
                analogReadBegin ();
                return;
        */

        rawSamples.push ( analogReadEnd () - soundMid ); // wait for previous conversion to finish and return the result
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

        // if ( rawSamples.overloaded () > 1 && rawSamples.overloaded () < 25 /* rawSamples.front () > soundTreshold */) {
        if (abs (rawSamples.front ()) >= soundTreshold && rawSamples.overloaded () == 0) {

                // output rawSamples
                        /*
                        cout << "SIGNAL (time = 7.1 ms)";
                        for (int i = 0; i < rawSamples.size (); i++)
                                cout << "," << rawSamples [i];
                        cout << endl;
                        */

                // calculate frequencies - FFT
                        complex<float> fftInput [rawSamplesCount];
                        complex<float> fftOutput [rawSamplesCount];
                        for (int i = 0; i < rawSamplesCount; i++)
                                fftInput [i] = { (float) rawSamples [i], 0.f };
                        fft (fftOutput, fftInput);

                // output frequencies
                        /*
                        cout << "MAGNITUDE (frequences up to = 17.87 kHz)";
                        for (int i = 0; i < rawSamplesCount / 2; i++)
                                cout << "," << abs (fftOutput [i]);
                        cout << endl;
                        */

                // calculate the sound features that will be the input to the neural network
                        float feature [featureCount] = {};
                        // 0. ZCR
                        for (int s = 1; s < rawSamples.size (); s++)
                                if (signbit (rawSamples [s - 1]) != signbit (rawSamples [s])) 
                                        feature [0] ++;
                        // 1. linear regression coeficient
                        int n = 0;
                        float sumX = 0;
                        float sumY = 0;
                        float sumXY = 0;
                        float sumX2 = 0;                
                        for (int i = 0; i < rawSamples.size (); i++) {                
                                float y = abs (rawSamples [i]);
                                float x = i;
                                n ++;
                                sumX += x;
                                sumY += y;
                                sumXY += x * y;
                                sumX2 += x * x;
                        }
                        feature [1] = (sumXY * n - sumX * sumY) / (sumX2 * n - sumX * sumX);
                        // 2, 3. energy in the first half and the second half of sound recording
                        for (int i = 0; i < rawSamples.size () / 2; i++) 
                                feature [2] += (float) rawSamples [i] * (float) rawSamples [i];
                        feature [2] = sqrt (feature [2] / rawSamples.size () / 2);
                        for (int i = rawSamples.size () / 2; i < rawSamples.size (); i++) 
                                feature [3] += (float) rawSamples [i] * (float) rawSamples [i];
                        feature [3] = sqrt (feature [3] / rawSamples.size () / 2);
                        // 4, 5, 6, 7. magnitudes in frequency bands
                        for (int i = 0; i < 5; i++) // 0 - 500 Hz
                                feature [4] += abs (fftOutput [i]);
                        for (int i = 5; i < 21; i++) // 500 Hz - 2.8 kHz
                                feature [5] += abs (fftOutput [i]);
                        for (int i = 21; i < 51; i++) // 2.8 - 7 kHz
                                feature [6] += abs (fftOutput [i]);
                        for (int i = 21; i < 51; i++) // 7 - 17.87 kHz
                                feature [7] += abs (fftOutput [i]);


                // ----- rule based decision: a clap or not a clap -----
                        /*
                        cout << "----- ZRC = " << feature [0] 
                        << " LRC = " << feature [1]
                        << " RMS = " << feature [2] << ", " << feature [3] // << " ratio: " << feature [3] / feature [2]
                        << " FREQ = " << feature [4] << ", " << feature [5] << ", " << feature [6] << ", " << feature [7] 
                        << " -----" << endl;
                        */

                        cout << "rule says: ";
                        bool err = false;
                        if (feature [0] < 13)                                                                                           { err = true; cout << " ZRC too low."; }
                        if (feature [0] > 39)                                                                                           { err = true; cout << " ZRC too high."; }
                        if (feature [1] > -0.4)                                                                                         { err = true; cout << " LRC too high."; }; 
                        if (feature [1] < -0.85)                                                                                        { err = true; cout << " LRC too low."; }; 
                        if (feature [3] / feature [2] < 0.3 || feature [3] / feature [2] > 0.7)                                         { err = true; cout << " Wrong RMS pattern."; }; 
                        if (feature [4] / feature [5] > 0.5 || feature [6] / feature [5] > 0.5 || feature [7] / feature [5] > 0.5)      { err = true; cout << " Wrong frequency pattern."; }; 
                        if (err) {
                                cout << " Not a clap." << endl;
                        } else {
                                cout << " it's a clap.\n";
                                digitalWrite (ruleBasedLedPin, !digitalRead (ruleBasedLedPin));
                        }

                // ----- neural network based decision: a clap or not a clap -----
                        // normalize ZCR
                        feature [0] /= 40; 
                        // normalize linear regression coeficient
                        feature [1] = -feature [1]; 
                        // normalize energy pattern
                        float sumRMS = feature [2] + feature [3];
                        feature [2] /= sumRMS;
                        feature [3] /= sumRMS;
                        // normalize frequency pattern
                        float sumFFT = feature [4] + feature [5] + feature [6] + feature [7];
                        feature [4] /= sumFFT;
                        feature [5] /= sumFFT;
                        feature [6] /= sumFFT;
                        feature [7] /= sumFFT;                        

                        float p = neuralNetwork.forwardPass (feature) [0];
                        if (p > 0.5) {
                                cout << "neural network says: it's a clap, p = " << p << endl;
                                digitalWrite (neuralNetworkBasedLedPin, !digitalRead (neuralNetworkBasedLedPin));
                        } else {
                                cout << "neural network says: not a clap, p = " << p << endl;
                        }

                // output feature pattern as C++ initializer list - these lists are needed for neural network training
                        /*
                        cout << "features = {";
                        for (size_t f = 0; f < featureCount; f++) {
                                if (f)
                                        cout << ",";
                                char buf [25];
                                #ifdef ARDUINO_ARCH_AVR // Assuming Arduino Mega or Uno
                                        dtostrf (feature [f], 1, 4, buf);
                                #else
                                        snprintf (buf, sizeof (buf), "%.4f", feature [f]);
                                #endif
                                cout << buf << "f";
                        }
                        cout << "}\n";                
                        */

                delay (100); // wait before reading the next clap
                rawSamples.clear ();
                analogReadEnd ();
                analogReadBegin ();
        }

    }
