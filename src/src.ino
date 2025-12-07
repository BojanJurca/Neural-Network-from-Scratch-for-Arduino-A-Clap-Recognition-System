/*

    Arduino_neuralNetwork.ino

    This file is part of Clap Recognition Using a Neural Network from Scratch (C++ for Arduino): https://github.com/BojanJurca/Neural-Network-from-Scratch-for-Arduino-A-Clap-Recognition-System

    Sound sampling and clap recognition using neural network on Arduino Mega 2560

    Bojan Jurca, Nov 26, 2025

*/


// platform abstraction 
#ifdef ARDUINO                  // Arduino build requires LightwaightSTL library: https://github.com/BojanJurca/Lightweight-Standard-Template-Library-STL-for-Arduino
    #include <array.hpp>
    #include <ostream.hpp>
//    #define rand() random(RAND_MAX)
    #define srand(X) randomSeed(X)
    #ifdef ARDUINO_ARCH_AVR     // Arduino AVR
        unsigned long time (void *p) { return millis (); } // introduce time function only for the purpose of srand (time (NULL)) would work on AVR boards as well
    #endif
#else                           // standard C++ build
    #include <cstddef>
    #include <array>
    #include <iostream>
    #include <iomanip>
    #include <cstring>
    using namespace std;
    #define PROGMEM             // compiles to nothin
    void setup (); void loop (); int main () { setup (); while (true) loop (); return 0; }
    #define cinit(...)          // compiles to nothing
    #define INPUT 1             // whatever
    #define OUTPUT 2            // whatever
    #define pinMode(...)        // compiles to nothing
    #define digitalRead(X) 0    // compiles to 0
    #define digitalWrite(...)   // compiles to nothing        
    #define delay(X)            // compiles to nothing
    unsigned char ADCSRA;
    #define ADPS0 0             // whatever
    #define ADPS1 1             // whatever
    #define ADPS2 2             // whatever
    #define ADSC 3              // whatever
    #define ADC 4               // whatever
    #define ADEN 5              // whatever
    #define REFS0 6             // whatever
    unsigned char ADMUX;
    #define bit(X) 0            // compiles to 0
    #define bitSet(X,Y)         // compiles to nothning
    #define bit_is_clear(X,Y) 0 // compiles to 0
#endif


// input: sampling definitions

    #define microphonePin 0             // the A/D pin you microphone is connected to the controller
    #define signalMid 511               // obtained by measuring the silence mean value of a particular microphone
    #define signalTreshold 350          // the treshold when the sound gets loud enough to indicate it may be a clap
    #define signalOverloaded 509        // obtained by observing the microphone signal - this is when the signal gets overloaded
    #define sampleCount 256             // 256 samples sampled with 35.75 kHz gives 7.16 ms of the sound
    #define samplingFrequency 35750.0f  // sampling frequency is determined by setting A/D prescale bits
    
// output: LED

    #define ledPin 2                    // the pin your LED is connected
   
// discrete Fourier transform

    #define NyquistFrequency ( samplingFrequency / 2 )
    #define distinctFftCoeficients  ((sampleCount + 1) / 2 + 1) // works for both, even and odd numbers

// mel filters and MFCCs, 19 and 13 give good results

    #define melFilterCount 19
    #define mfccCount 13        // must be < melFilterCount, tipically 12 or 13

// sound pattern features

    #define ZCR 1 // 1 - use this feature, 0 - don't use this feature
    #define LRC 1 // 1 - use this feature, 0 - don't use this feature

    #define featureCount ( ZCR + LRC + mfccCount ) // feature extracted from sound recordings as the input to the neural network
    // feature [0]              <- zerro crossings - the number the signal crosses time axes
    // feature [1]              <- linear regression coeficient - how fast the signal amplitude is dropping    
    // feature [2, ...]         <- MFCC [0], ...

// void extractFeaturesFromSoundRecording (float feature [featureCount], int soundRecording [sampleCount]);

    #include "dsp.h"

// ----- the neural network -----
 
    #include "neuralNetwork.hpp"
    
    //                                  .--- the number of inputs (= the number of pattern features)
    //                                  |          .--- first layer activation function
    //                                  |          |      .--- the number of neurons in the firt layer
    //                                  |          |      |                                      .--- output layer activation function
    //                                  |          |      |                                      |     .--- the number of neurons in the output layer - it corresponds to the number of categories that the neural network recognizes (clap or not a clap in our case)
    //                                  |          |      |                                      |     |
    typedef neuralNetworkLayer_t<featureCount, FastTanh, 11, /* add more layers if needed */ FastTanh, 2> neuralNetwork_t; // this configuration gives good results but try others as well  
    // at this point neuralNetwork is initialized with random weights and biases and it is ready for training
    // - you can either start training it and export the trained model when fiished
    // - or you can load already trained model that is cappable of making usable outputs 

    // define neural network as a union with array of floats, with this little trick we'll keep only one copy of initializing data in RAM
    #ifdef ARDUINO  // initialize neural network model without calling the constructor that would randomize it - neural network won't do the training, so randomization is not needed
        static const union {
            float model [sizeof (neuralNetwork_t) / sizeof (float)];
            neuralNetwork_t n;
        } n =  {    -0x1.caa234p-1f,-0x1.295666p-2f,-0x1.13a38cp-3f,0x1.297e98p+2f,0x1.d1894cp+1f,0x1.0c2604p+2f,0x1.055c5ep+0f,0x1.3df486p-1f,-0x1.3a8d54p+0f,-0x1.a92e78p+0f,
                    0x1.c61922p+0f,0x1.236364p-1f,-0x1.778752p+0f,0x1.2a12b4p-3f,0x1.606752p+0f,-0x1.008c92p+2f,0x1.345314p+0f,-0x1.2eb5bap+1f,0x1.0097b6p+2f,0x1.0ce6cap+1f,
                    0x1.9fcc3ep-1f,0x1.3c3a84p+0f,0x1.599472p-1f,-0x1.80ab9p-1f,0x1.9aef7cp-1f,-0x1.a956ecp-1f,-0x1.a72332p+0f,-0x1.49c57p-1f,-0x1.44ca58p+1f,-0x1.2a1d7cp-2f,
                    -0x1.ffe7d8p+0f,0x1.4269d6p+0f,0x1.699b0ep+0f,0x1.658958p-6f,0x1.56fd1p+0f,-0x1.8bcc3p+1f,-0x1.3d4bdap-2f,0x1.8b3f6cp-2f,0x1.65f874p+0f,-0x1.0c070cp+0f,
                    0x1.14ffacp+1f,0x1.2db9c8p+1f,0x1.7e12d6p+0f,-0x1.21ed44p-2f,-0x1.3cf888p+2f,0x1.2713dcp-4f,0x1.26b794p+1f,-0x1.63965ap+0f,-0x1.47987ap+0f,0x1.4f5402p+0f,
                    -0x1.969256p+1f,-0x1.aa932ep-1f,0x1.ebc344p+0f,-0x1.2e2feep+0f,0x1.046e1ep-1f,-0x1.58787cp-1f,-0x1.e061ecp-2f,-0x1.4f263p-3f,-0x1.391382p-3f,-0x1.0fd624p+1f,
                    0x1.b52adp-4f,0x1.0f2452p+1f,-0x1.02f17ep+0f,-0x1.0fb186p+0f,0x1.3c9484p+0f,-0x1.88976p+0f,0x1.fa768ep-2f,0x1.1bf23cp-1f,-0x1.a70f96p-1f,-0x1.effcccp-1f,
                    -0x1.c062e6p-2f,-0x1.189f72p-3f,-0x1.71acc4p+0f,0x1.cb2572p-4f,-0x1.479eb8p+1f,0x1.47937ap+0f,0x1.c2485ap+0f,0x1.8a5f96p+0f,-0x1.862cb2p+0f,0x1.2d893cp+1f,
                    -0x1.41e656p+1f,-0x1.dd26cep+0f,0x1.85dd58p+0f,0x1.bf7a4ep-3f,-0x1.6b3c94p+0f,0x1.2024b6p-1f,-0x1.14b9ep+1f,0x1.7b1ee4p+0f,-0x1.0467f8p+1f,-0x1.24c6c8p-1f,
                    -0x1.3541fp+1f,-0x1.75114ep+1f,0x1.41fb9cp+1f,0x1.14e81cp+1f,0x1.8fdad8p+2f,0x1.6e39p+2f,0x1.e06068p+1f,-0x1.00806ap+1f,-0x1.de8842p+1f,-0x1.10e34ep+1f,
                    -0x1.163aep-2f,-0x1.658024p-2f,0x1.a0b49cp+2f,0x1.90a6fap+2f,0x1.f978b4p-3f,-0x1.3109e4p+0f,-0x1.6fa6fcp-5f,0x1.e139e8p-1f,-0x1.642ce8p-5f,-0x1.6ee1bep+0f,
                    0x1.2182b6p+1f,0x1.6418a6p-2f,-0x1.6ffdaep-1f,-0x1.7db01cp-2f,0x1.417f0cp-1f,-0x1.e079f6p-1f,-0x1.7b984ap-1f,0x1.4ed276p-1f,-0x1.11d1acp-2f,-0x1.066a9ap+0f,
                    0x1.468a6p-3f,0x1.d5b66ep+0f,0x1.1a8906p-4f,-0x1.2bb06ap-1f,0x1.983756p-3f,-0x1.ad80b6p-2f,-0x1.9911p+1f,0x1.214944p-4f,0x1.fffaf8p-3f,-0x1.68b92ep+1f,
                    -0x1.003a1p+0f,0x1.df4baep-1f,0x1.871a2cp+0f,0x1.02aaecp+0f,0x1.a48da8p+0f,-0x1.6608f8p+1f,-0x1.c501a6p-2f,-0x1.209ac6p-1f,-0x1.108278p+1f,-0x1.e3d8ap+0f,
                    0x1.e70d6p-2f,0x1.1a2562p+0f,-0x1.47fdbap+1f,-0x1.ec4dfp+1f,-0x1.d200d4p-1f,0x1.3ebcfp+0f,0x1.ae5564p+0f,0x1.2868bp+1f,0x1.163eecp+1f,0x1.8a1b82p+0f,
                    0x1.0f37a8p+0f,-0x1.aa6f06p+0f,-0x1.63003ep-2f,0x1.0da116p-2f,0x1.66312ep+0f,0x1.213c98p+0f,-0x1.e644bp-1f,0x1.717cp-1f,0x1.6901fap+0f,0x1.45e23p-1f,
                    0x1.24ef9ep+0f,0x1.fd5c02p-1f,-0x1.56b45cp+0f,-0x1.f4c0dap-4f,-0x1.d3b06ep-1f,0x1.3ee04ep+1f,0x1.2abcfcp+0f,0x1.a199d6p+0f,0x1.fcbd66p-2f,0x1.81ac14p+0f,
                    -0x1.7d00b6p-2f,0x1.c62364p+2f,-0x1.b073e6p-1f,0x1.0bd5aap+0f,-0x1.906684p-1f,-0x1.0aaf3cp+0f,0x1.1c965ap+1f,-0x1.523a94p-1f,0x1.6c76c8p+1f,-0x1.aa32cap+1f,
                    -0x1.045c08p+1f,0x1.9b519cp+1f,-0x1.3c9a98p+1f,0x1.f795eep-5f,-0x1.5f45a6p+1f,0x1.9a4638p+1f,-0x1.095278p+2f,-0x1.9b170ep+1f,0x1.16b6ecp+1f,-0x1.b663cep+1f,
                    -0x1.6dca1p-1f,0x1.0635acp+1f,-0x1.9492eep+0f,0x1.2012d4p+1f,-0x1.affd0ep+1f,-0x1.b0120ap-3f,-0x1.039ba8p-7f,0x1.2b64c2p-1f,0x1.0a59a6p+1f,0x1.334b2p+1f    };
    #else // with standard C++ call the constructor that would randomize it - needed for training
        static struct { // pack the neural network into structure so it can be called the same way as in "union" case (n.n)
            neuralNetwork_t n;
        } n;
    #endif

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


        #ifndef ARDUINO     // training is too demanding for Arduino and needs to be done a a computer

            cout << "----- TRAINING BEGIN (better to use a computer instead of a controller) -----\n";
            
            // Split all recordings into three sets:
            //
            // 1. Training set (~80%):
            // 2. Validation set (~10%):
            // 3. Test set (~10%):
            
            #define SHIFT_SETS 1
            #define TRAINING_SET ((p + SHIFT_SETS)) % 9 <= 6
            #define VALIDATION_SET ((p + SHIFT_SETS)) % 9 == 7
            #define TEST_SET ((p + SHIFT_SETS)) % 9 == 8
            
            // train the neural network with training patterns, this easier done on bigger computers, it doesn't have to be an Arduino
            #include "trainingRecordings.h"
            

            // store the best training result
            float lowestLoss = 1.0f / 0.0f;
            decltype (n.n) lowestLossModel = n.n;
            float highestAccuracy = 0;
            float lowestLossAtHighestAccuracy = lowestLoss;
            decltype (n.n) highestAccuracyModel = n.n;

            // extract features (patterns) from clap audio_recordings
            //float clapFeatures [clapRecordingCount][featureCount];
            array<float, featureCount> clapFeatures [recordingCount];
            for (int i = 0; i < clapRecordingCount; i++)
                // extractFeaturesFromSoundRecording (clapFeatures [i], clapRecording [i]);
                clapFeatures [i] = extractFeatures (clapRecording [i]);

            // extract features (patterns) from other audio_recordings
            //float otherFeatures [recordingCount][featureCount];
            array<float, featureCount> otherFeatures [recordingCount];
            for (int i = 0; i < recordingCount; i++)
                //extractFeaturesFromSoundRecording (otherFeatures [i], otherRecording [i]);
                otherFeatures [i] = extractFeatures (otherRecording [i]);


            // do, say 20 independent gradient descents starting from different random initializations
            for (int d = 0; d < 10; d ++) {

                // do, say, 50000 (epoch = 50000) gradient descent iterations to reach a local minimum
                for (int e = 0; e < 50000; e ++) {
                    float loss = 0.0f;

                    // gradient descent
                    for (int p = 0; p < recordingCount; p ++) {
                        if ( TRAINING_SET ) {
                            loss += n.n.backwardPropagation (clapFeatures [p], {1, 0});   // index 0 = clap
                            loss += n.n.backwardPropagation (otherFeatures [p], {0, 1});  // index 1 = other sound
                        }
                    }    
                    // keep track of the model that gives the lowest loss           
                    if (loss < lowestLoss) {
                        lowestLoss = loss;
                        lowestLossModel = n.n;
                    } 
                    
                    // occasionaly output and validate the intermediate results 
                    if (e % 1000 == 999) {
                        // validation
                        int clapsRecognized = 0;
                        int clapsMissed = 0;

                        for (int p = 0; p < recordingCount; p++)
                            if ( VALIDATION_SET )
                                if (softmax (n.n.forwardPass (clapFeatures [p])) [0] > 0.5) // if clap probability > 0.5
                                    clapsRecognized ++;
                                else
                                    clapsMissed ++;
                        float clapAccuracy = (float) clapsRecognized / (clapsRecognized + clapsMissed);
                            
                        int otherRecognized = 0;
                        int otherMissed = 0;
                                        
                        for (int p = 0; p < recordingCount; p ++)
                            if ( VALIDATION_SET )
                                if (softmax (n.n.forwardPass (otherFeatures [p])) [1] > 0.5) // if other sound probability > 0.5
                                    otherRecognized ++;
                                else
                                    otherMissed ++;
                        float otherAccuracy = (float) otherRecognized / (otherRecognized + otherMissed);

                        float accuracy = ((float) clapsRecognized + (float) otherRecognized) / ((float) clapsRecognized + (float) otherRecognized + (float) clapsMissed + (float) otherMissed);
                        if (accuracy > highestAccuracy) {
                            highestAccuracy = accuracy;
                            highestAccuracyModel = n.n;
                            lowestLossAtHighestAccuracy = loss;
                        }
                        if (loss < lowestLossAtHighestAccuracy && accuracy == highestAccuracy) {
                             lowestLossAtHighestAccuracy = loss;
                             highestAccuracyModel = n.n;
                        }

                        cout << "   d: " << d << "   e: " << e << "   loss: " << loss << "   lowest loss: " << lowestLoss << "   accuracy: " << accuracy * 100 << "% (claps: " << clapAccuracy * 100 << "%   others: " << otherAccuracy * 100 << "%)   highest accurcy: " << highestAccuracy * 100 << "% (loss: " << lowestLossAtHighestAccuracy << ")" << endl;
                    }
                }           

                // start the next gradient descent with different random seed
                srand (static_cast<unsigned> (time (nullptr)));
                n.n.randomize ();
            }
            
            cout << "----- TRAINING END ------\n";
            cout << "----- TESTING BEGIN -----\n";

            int clapsRecognized = 0;
            int clapsMissed = 0;
            for (int p = 0; p < recordingCount; p++)
                if ( TEST_SET )
                    if (softmax (lowestLossModel.forwardPass (clapFeatures [p])) [0] > 0.5) // if clap probability > 0.5
                        clapsRecognized ++;
                    else
                        clapsMissed ++;
            float clapAccuracy = (float) clapsRecognized / (clapsRecognized + clapsMissed);
            int otherRecognized = 0;
            int otherMissed = 0;
            for (int p = 0; p < recordingCount; p ++)
                if ( TEST_SET )
                    if (softmax (lowestLossModel.forwardPass (otherFeatures [p])) [1] > 0.5) // if other sound probability > 0.5
                        otherRecognized ++;
                    else
                        otherMissed ++;
            float otherAccuracy = (float) otherRecognized / (otherRecognized + otherMissed);
            float accuracy = ((float) clapsRecognized + (float) otherRecognized) / ((float) clapsRecognized + (float) otherRecognized + (float) clapsMissed + (float) otherMissed);

            cout << "lowest loss model accuracy: " << accuracy * 100 << "% (claps: " << clapAccuracy * 100 << "%   others: " << otherAccuracy * 100 << "%)" << endl;
            cout << "lowest loss model = " << hexfloat << lowestLossModel << defaultfloat << endl;

            clapsRecognized = 0;
            clapsMissed = 0;
            for (int p = 0; p < recordingCount; p++)
                if ( TEST_SET )
                    if (softmax (highestAccuracyModel.forwardPass (clapFeatures [p])) [0] > 0.5) // if clap probability > 0.5
                        clapsRecognized ++;
                    else
                        clapsMissed ++;
            clapAccuracy = (float) clapsRecognized / (clapsRecognized + clapsMissed);
            otherRecognized = 0;
            otherMissed = 0;
            for (int p = 0; p < recordingCount; p ++)
                if ( TEST_SET )
                    if (softmax (highestAccuracyModel.forwardPass (otherFeatures [p])) [1] > 0.5) // if other sound probability > 0.5
                        otherRecognized ++;
                    else
                        otherMissed ++;
            otherAccuracy = (float) otherRecognized / (otherRecognized + otherMissed);
            accuracy = ((float) clapsRecognized + (float) otherRecognized) / ((float) clapsRecognized + (float) otherRecognized + (float) clapsMissed + (float) otherMissed);

            cout << "-------------------------\n\n";
            
            cout << "highest accuracy model accuracy: " << accuracy * 100 << "% (claps: " << clapAccuracy * 100 << "%   others: " << otherAccuracy * 100 << "%)" << endl;
            cout << "highest accuracy model = " << hexfloat << highestAccuracyModel << defaultfloat << endl;

            cout << "----- TESTING END -------\n";

            // this usually is a better choice
            n.n = highestAccuracyModel;

        #endif

        
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


        #ifndef ARDUINO         //  standard C++ build (training only)
            exit (0);
        #endif
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
           
            //float feature [featureCount];
            //extractFeaturesFromSoundRecording (feature, soundRecording);
            array<float, featureCount> feature;
            feature = extractFeatures (soundRecording);
            /*
                for (int i = 0; i < featureCount; i++)
                    cout << feature [i] << endl;
            */


            // ask neural network what it thinks about these features
            float p = softmax (n.n.forwardPass (feature)) [0];
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