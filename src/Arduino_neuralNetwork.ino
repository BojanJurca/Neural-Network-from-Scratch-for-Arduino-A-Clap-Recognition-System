/*

    Arduino_neuralNetwork.ino

    This file is part of Clap Recognition Using a Neural Network from Scratch (C++ for Arduino): https://github.com/BojanJurca/Neural-Network-from-Scratch-for-Arduino-A-Clap-Recognition-System

    Sound sampling and clap recognition using neural network on Arduino Mega 2560

    Bojan Jurca, Sep 9, 2025

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
    #define distinctFftCoeficients ( sampleCount / 2 + 1 )

// mel filters

    #define melFilterCount 6 // it seems that 6 gives good results but try different numbers as well

// sound pattern features

    #define featureCount ( 2 + melFilterCount ) // feature extracted from sound recordings as the input to the neural network
    // feature [0]              <- zerro crossings - the number the signal crosses time axes
    // feature [1]              <- linear regression coeficient - how fast the signal amplitude is dropping    
    // feature [2, ...]         <- mel filters

// include a small compatibility.h library with which the same code will run on Arduino Mega 2560 and desktop computer as well

    #include "compatibility.h" // Arduino <-> standard C++ compatibility

// void extractFeaturesFromSoundRecording (float feature [featureCount], int soundRecording [sampleCount]);

    #include "dsp.h"


// ----- the neural network -----
 
    #include "neuralNetwork.hpp"
    
    //                    .--- the number of neurons in the first layer - the number of input sound features in our case
    //                    |                .--- second layer activation function
    //                    |                |    .--- the number of neurons the second layer 
    //                    |                |    |                             .--- output layer activation function
    //                    |                |    |                             |      .--- the number of neurons in the output layer - it corresponds to the number of categories that the neural network recognizes (clap or not a clap in our case)
    //                    |                |    |                             |      |
    neuralNetworkLayer_t<featureCount, Sigmoid, 5, /* add more if needed */ Sigmoid, 2> neuralNetwork; // this topology gives good results but try other topologies as well 
    
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
                // import trained model

                cout << "importing trained model\n";

                const int32_t model [] PROGMEM = { 1088934982,1056125554,1060564156,-1087099479,-1066412161,-1074884735,-1067327234,1057867316,1094637831,-1090546581,-1067851532,1075940152,1073899499,1065228311,1079703855,-1060683450,
                                                   1082809745,1059553803,1076080741,-1080144655,-1069457469,-1073204993,-1064810862,1071951074,-1069875473,1028319942,1074040879,-1082136736,-1073597770,-1079754927,-1076530130,1076574819,
                                                   -1067667764,1033151989,1062593514,-1085613887,-1080690817,-1090659372,-1080401428,1071313246,1043351735,-1063435984,1063655380,1070507908,1065360061,-1059851737,1091164282,-1063203450,
                                                   -1069429137,-1069139415,1086766935,-1056190823,1085179138,1079357277,1072314658,-1081688677,1067143074};
               neuralNetwork = model;
                
        /**/
        
        /*
            // train the neural network with training patterns, this is better done on bigger computer, 
            // it doesn't have to be an Arduino
            
            #include "trainingRecordings.h"
            
            cout << "training the neural network\n";

                // extract features from training audio_recordings
                
                float clapFeatures [clapRecordingCount][featureCount];
                
                cout << "    extracting " << featureCount << " features from " << clapRecordingCount << " claps\n";
                
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

                float otherFeatures [otherRecordingCount][featureCount];
                
                cout << "    extracting " << featureCount << " features from " << otherRecordingCount << " other sounds\n";
                
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

                constexpr int setCount = min (clapRecordingCount, otherRecordingCount);

                cout << "        " << setCount << " of clap feature patterns\n";
                cout << "        " << setCount << " of other sound feature patterns\n";
                cout << "    roughly 80 % of them will be used for training and roughly 20 % for testing\n\n";

                // This part, including testing different typologies, can be done more efficiently on larger computers and not necessarily on a controller,
                // as Arduino code is portable to standard C++.

                #define epoch 6201 // choose the right number of training iterations so the model gets trained but not overtrained
                for (int trainingIteration = 0; trainingIteration < epoch; trainingIteration++) {
                        float errorOverAllPatterns = 0;
                        
                        for (size_t i = 0; i < setCount; i++) {
                                if (i % 5 != 0) { // leave every 5th pattern for testing
                                        errorOverAllPatterns += neuralNetwork.backwardPropagation (clapFeatures [i], {1, 0});   // index 0 = clap
                                        errorOverAllPatterns += neuralNetwork.backwardPropagation (otherFeatures [i], {0, 1});  // index 1 = another sound
                                }
                        }

                        if (trainingIteration % 100 == 0) {
                            // test success
                            int clapsTested = 0;
                            int clapsRecognized = 0;
                            for (size_t i = 0; i < setCount; i++) 
                                    if (i % 5 == 0) { // take every 5th pattern for testing
                                        clapsTested ++;
                                        auto probability = neuralNetwork.forwardPass (clapFeatures [i]);
                                        if (probability [0] > 0.5) // index 0 = clap
                                            clapsRecognized ++;
                                    }
                            int otherTested = 0;
                            int otherRecognized = 0;
                            for (size_t i = 0; i < setCount; i++) 
                                    if (i % 5 == 0) { // take every 5th pattern for testing
                                        otherTested ++;
                                        auto probability = neuralNetwork.forwardPass (otherFeatures [i]);
                                        if (probability [1] > 0.5) // index 1 = other sound
                                            otherRecognized ++;
                                    }                            
                                        
                            cout << "    iteration " << trainingIteration << "    error over all patterns = " << errorOverAllPatterns << "   classification accuracy = " << (float) (clapsRecognized +  otherRecognized) / (clapsTested + otherTested) * 100 << "   (claps " << (float) clapsRecognized / clapsTested * 100 << " %   other sounds " << (float) otherRecognized / otherTested * 100 << " %) " << endl;
                            // cout << trainingIteration << "   " << errorOverAllPatterns << "   " << (float) (clapsRecognized +  otherRecognized) / (clapsTested + otherTested) * 100 <<endl;
                        }
                }


                // export trained model
                cout << "\ntrained model = " << neuralNetwork << endl;
        */
        

        cout << "clap recognition system is ready\n\n";

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