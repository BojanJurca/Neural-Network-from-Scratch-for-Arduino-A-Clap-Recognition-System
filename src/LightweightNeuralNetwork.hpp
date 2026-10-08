/*

    LightweightNeuralNetwork.hpp

    This file is part of Clap Recognition Using a Neural Network from Scratch (C++ for Arduino): https://github.com/BojanJurca/Neural-Network-from-Scratch-for-Arduino-A-Clap-Recognition-System

    The neural network is implemented as a C++ variadic template. This approach was chosen to simplify both the usage 
    of the library and the modification of the network topology, making it as intuitive and flexible as possible.
 

        //                    .--- the number of inputs
        //                    |    .--- the first layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
        //                    |    |    .--- the number of neurons the first layer 
        //                    |    |    |                                      .--- output layer activation function (Sigmoid, ReLU, Tanh, FastTanh)
        //                    |    |    |                                      |      .--- the number of neurons in the output layer = the number of outputs
        //                    |    |    |                                      |      |
        neuralNetworkLayer_t<13, Tanh, 16, / * add more layers if needed * / Sigmoid, 2> neuralNetwork;


    Bojan Jurca, Oct 23, 2026

*/


#ifndef __NEURAL_NETWORK_HPP__
    #define __NEURAL_NETWORK_HPP__


    // ----- platform abstraction  -----
    
    #ifdef ARDUINO                  // Arduino build requires LightwaightSTL library: https://github.com/BojanJurca/Lightweight-Standard-Template-Library-STL-for-Arduino
        #include <array.hpp>
        #include <ostream.hpp>
        #define rand () random (RAND_MAX)
    #else                           // standard C++ build
        #include <array>
        #include <iostream>
        #include <cstring>
        #include <cmath>
        using namespace std;
        #include <cstdint>
    #endif
    #include <float.h>


    // ----- non-linear neuron activation functions and their derivatives -----

        inline __attribute__((always_inline)) float __sigmoid__ (float x) {
            return 1.f / (1.f + exp (-x));
        }
        inline __attribute__((always_inline)) float __sigmoidDerivative__ (float x) {
            float s = __sigmoid__ (x);
            return s * (1 - s);
        }
        
        // sigmoid approximation for faster calculation
        inline __attribute__((always_inline)) float __fastSigmoid__ (float x) {
            return 0.5f * (x / (1.0f + fabs (x)) + 1.0f);
        }
        inline __attribute__((always_inline)) float __fastSigmoidDerivative__ (float x) {
            float t = 1.0f + fabsf (x);
            return 0.5f / (t * t);
        }

        inline __attribute__((always_inline)) float __ReLU__ (float x) {
            return x <= 0.f ? 0.f : x;
        }
        inline __attribute__((always_inline)) float __ReLUDerivative__ (float x) {
            return x <= 0.f ? 0.f : 1.f;
        }

        inline __attribute__((always_inline)) float __tanh__ (float x) {
            float ex = expf (x);
            float enx = expf (-x);
            return (ex - enx) / (ex + enx);
        }
        inline __attribute__((always_inline)) float __tanhDerivative__ (float x) {
            float ex = expf (x);
            float enx = expf (-x);
            float t = (ex - enx) / (ex + enx);
            return 1.0f - t * t;
        }

        // tanh' approximation for faster calculation
        inline __attribute__((always_inline)) float __fastTanh__ (float x) {
            if (x < -3) return -1;
            if (x > 3) return 1;
            float x2 = x * x;
            return x * (27 + x2) / (27 + 9 * x2);
        }
        inline __attribute__((always_inline)) float __fastTanhDerivative__ (float x) {
            float ex = expf (x);
            float enx = expf (-x);
            float t = (ex - enx) / (ex + enx);
            return 1.0f - t * t;
        }

        static constexpr size_t Sigmoid      = 0;
        static constexpr size_t FastSigmoid  = 1;
        static constexpr size_t ReLU         = 2;
        static constexpr size_t Tanh         = 5;
        static constexpr size_t FastTanh     = 6;

        // neuron activation function (Sigmoid, FastSigmoid, ReLU, Tanh or FastTanh)
        template <size_t activationFunction>
        inline __attribute__((always_inline)) float af (float x) {
                 if (activationFunction == Sigmoid)     return __sigmoid__ (x); 
            else if (activationFunction == FastSigmoid) return __fastSigmoid__ (x);
            else if (activationFunction == ReLU)        return __ReLU__ (x);
            else if (activationFunction == Tanh)        return __tanh__ (x);
            else if (activationFunction == FastTanh)    return __fastTanh__ (x);
            // fallback, shouldn't happen
            // static_assert (true, "Unsupported activation function");
            return 0.f; // fallback, shouldn't happen
        }
        template <size_t activationFunction>
        float afDerivative (float x) {
                 if (activationFunction == Sigmoid)     return __sigmoidDerivative__ (x);
            else if (activationFunction == FastSigmoid) return __fastSigmoidDerivative__ (x); 
            else if (activationFunction == ReLU)        return __ReLUDerivative__ (x);
            else if (activationFunction == Tanh)        return __tanhDerivative__ (x);
            else if (activationFunction == FastTanh)    return __fastTanhDerivative__ (x);
            // fallback, shouldn't happen
            // static_assert (false, "Unsupported activation function");
            return 0.f; 
        }

        template <size_t activationFunction>
        inline __attribute__((always_inline)) const char *afName () {
                 if (activationFunction == Sigmoid)     return "Sigmoid";
            else if (activationFunction == FastSigmoid) return "FastSigmoid"; 
            else if (activationFunction == ReLU)        return "ReLU";
            else if (activationFunction == Tanh)        return "Tanh";
            else if (activationFunction == FastTanh)    return "FastTanh";
            // fallback, shouldn't happen
            // static_assert (false, "Unsupported activation function");
            return "Error"; 
        }


    // ----- random initializer -----

        template <size_t activationFunction>
        float randomInitializer (size_t inputCount, size_t neuronCount) {
            // use Box-Muller Transform to calculate normaly distributed random variable from uniformly distrubuted random function
        
                // select 2 independent uniformly distributed random values in interval (0, 1)
                float U1 = ((float) rand () + 1.0f) / ((float) RAND_MAX + 2.0f);
                float U2 = ((float) rand () + 1.0f) / ((float) RAND_MAX + 2.0f);
                
                // use Box-Muller to tranform them to two independed normally distributed random values with mean of 0 and variance of 1
                float N1 = sqrt (-2 * log (U1)) * cos (2 * M_PI * U2); 
                // float N2 = sqrt (-2 * log (U1)) * sin (2 * M_PI * U2); // we don't actually need the second independent random variable here
        
            // apply the desired mean and variance
                     if (activationFunction == Sigmoid)     return 0 + sqrt (2.0 / (inputCount + neuronCount)) * N1; // Xavier (Glorot)
                else if (activationFunction == FastSigmoid) return 0 + sqrt (2.0 / (inputCount + neuronCount)) * N1; // Xavier (Glorot)
                else if (activationFunction == ReLU)        return 0 + sqrt (2.0 / inputCount) * N1;                 // He
                else if (activationFunction == Tanh)        return 0 + sqrt (2.0 / (inputCount + neuronCount)) * N1; // Xavier (Glorot)
                else if (activationFunction == FastTanh)    return 0 + sqrt (2.0 / (inputCount + neuronCount)) * N1; // Xavier (Glorot)
                // fallback, shouldn't happen
                // static_assert (activationFunction < 6, "Unsupported activation function");
                return sqrtf (1.0f / inputCount) * N1;
        }


    // ----- neural network -----

    // training

        // learning rate
        #ifndef learningRate
            #define learningRate 0.01
        #endif


    // basic neuralNetwork_t class template, not used but needed by C++ compiler
        template <size_t... sizes> 
        class neuralNetworkLayer_t;


    // hidden layers
        template <size_t inputCount, size_t activationFunction, size_t neuronCount, size_t... sizes> 
        class neuralNetworkLayer_t<inputCount, activationFunction, neuronCount, sizes...> {

                // data structures needed for this layer: weight and bias
                float weight [neuronCount][inputCount];
                float bias [neuronCount];

                // include the next layer instance which will include the next layer itself, ...
                neuralNetworkLayer_t<neuronCount, sizes...> nextLayer;

            public:
            
                static constexpr size_t outputCount = neuralNetworkLayer_t<neuronCount, sizes...>::outputCount;
                using output_t = array<float, outputCount>;

                // calculates the neurons of this layer and returns the category that the input belongs to
                template<typename input_t>
                output_t forwardPass (const input_t (&input) [inputCount]) const {   
                    float neuron [neuronCount];

                    // neuron = af (w x input + bias)
                        for (size_t n = 0; n < neuronCount; n++) {
                            neuron [n] = bias [n];
                            for (size_t i = 0; i < inputCount; i++)
                                neuron [n] += weight [n][i] * input [i];
                            neuron [n] = af<activationFunction> (neuron [n]);
                            // cout << "   hidden layer neuron [" << n << "] = " << neuron [n] << endl;
                        }

                    // return what the next layer thinks about the neurons clculated here
                        return nextLayer.forwardPass (neuron);
                }
            
                // make it possible to use arrays instead of C arrays
                template<typename input_t>
                __attribute__((always_inline))
                inline output_t forwardPass (const array<input_t, inputCount> input) const {   
                    return forwardPass (*reinterpret_cast<const input_t (*)[inputCount]> (input.data ()));
                }


            // training

                // layer_t constructor - initialization of weight and bias
                neuralNetworkLayer_t () {
                    randomizeLayer ();
                }
                
                void randomizeLayer () {
                    for (size_t n = 0; n < neuronCount; n++) {
                        for (size_t i = 0; i < inputCount; i++)
                            // weight are typically initialized with random numbers, He function is is particularly suitable to be used with ReLU activation function
                            weight [n][i] = randomInitializer<activationFunction> (inputCount, neuronCount);
                        // bias are typically set to 0 at initialization
                        bias [n] = 0;
                    }
                }
                
                void randomize () {
                    randomizeLayer ();
                    nextLayer.randomizeLayer ();
                }


                // iterate from the last layer to the first and adjust weight and bias meanwhile, returns the error clculated at output layer
                template<typename input_t, typename expected_t>
                float backwardPropagation (const input_t (&input) [inputCount], const expected_t (&expected) [outputCount], float previousLayerDelta [inputCount] = NULL) { // the size of expected in all layers equals the size of the output of the output layer

                    // while moving forward do exactly the same as forwardPass function does
                        float z [neuronCount];
                        float neuron [neuronCount];

                        // z = weight x input + bias
                        // neuron = af (z)
                        for (size_t n = 0; n < neuronCount; n++) {
                            z [n] = bias [n];
                            for (size_t i = 0; i < inputCount; i++)
                                z [n] += weight [n][i] * input [i];
                            neuron [n] = af<activationFunction> (z [n]);
                        }

                    // calculate the first part of delta in the next layer then apply activation function derivative here
                        // delta = next layer weight * next layer delta * a' (z)
                        float delta [neuronCount]; 
                        float error = nextLayer.backwardPropagation (neuron, expected, delta);
                        // calculate only the second part of delta, the first par has already been calculated at the next layer
                        for (size_t n = 0; n < neuronCount; n++) {
                            delta [n] *= afDerivative<activationFunction> (z [n]);
                            // cout << "   hidden layer delta [" << n << "] = " << delta [n] << endl;
                        }

                    // update weight and bias at this layer
                        for (size_t n = 0; n < neuronCount; n++) {

                            // update weight
                            for (size_t i = 0; i < inputCount; i++)
                                weight [n][i] -= learningRate * delta [n] * input [i];

                            // update bias
                            bias [n] -= learningRate * delta [n];
                        }

                    // calculate only the first part of previous layer delta, since z from previous layer is not available here
                        // previousLayerDelta = weight * delta * a' (previous layer z)
                        if (previousLayerDelta) {
                            for (size_t i = 0; i < inputCount; i++) {
                                previousLayerDelta [i] = 0;
                                for (size_t n = 0; n < neuronCount; n++)
                                    previousLayerDelta [i] += weight [n][i] * delta [n];
                            }
                        }

                    return error;
                }

                // make it possible to use arrays instead of C arrays
                template<typename input_t, typename expected_t>
                __attribute__((always_inline))
                inline float backwardPropagation (const array<input_t, inputCount> input, const expected_t (&expected) [outputCount], float previousLayerDelta [inputCount] = NULL) {
                    return backwardPropagation (*reinterpret_cast<const input_t (*)[inputCount]> (input.data ()), expected, previousLayerDelta);
                }

                template<typename input_t, typename expected_t>
                __attribute__((always_inline))
                inline float backwardPropagation (const input_t (&input) [inputCount], const array<expected_t, outputCount> expected, float previousLayerDelta [inputCount] = NULL) {
                    return backwardPropagation (input, *reinterpret_cast<const expected_t (*)[outputCount]> (expected.data ()), previousLayerDelta);
                }

                template<typename input_t, typename expected_t>
                __attribute__((always_inline))
                inline float backwardPropagation (const array<input_t, inputCount> input, const array<expected_t, outputCount> expected, float previousLayerDelta [inputCount] = NULL) {
                    return backwardPropagation (*reinterpret_cast<const input_t (*)[inputCount]> (input.data ()), *reinterpret_cast<const expected_t (*)[outputCount]> (expected.data ()), previousLayerDelta);
                }

                // export the whole model as C++ initializer list
                friend ostream& operator << (ostream& os, const neuralNetworkLayer_t& nn) {
                    float *p = (float *) &nn;
                    for (size_t i = 0; i < sizeof (nn) / sizeof (float); i++) {
                        if (i > 0) {
                            os << ",";
                            if (i % 10 == 0)
                                os << endl;
                        }
                        os << *(p + i) << 'f';
                    }
                    return os;
                }

                // export visualized model as C++ initializer list
                struct visualized_t { const neuralNetworkLayer_t& layer; };
                visualized_t visualize () const { return { *this }; }
                friend ostream& operator << (ostream& os, const visualized_t& v) {
                    float Min = FLT_MAX;
                    float Max = -FLT_MAX;

                    os << "   // ----- layer inputs: " << inputCount << ", outputs (neurons): " << neuronCount << ", activation: " << afName<activationFunction> () << " -----\n";
                    os << "   //    --- weights ---\n";
                    for (size_t n = 0; n < neuronCount; n++) {
                        os << "         ";
                        for (size_t i = 0; i < inputCount; i++) {
                            os << v.layer.weight [n][i] << "f, ";
                            if (v.layer.weight [n][i] < Min) Min = v.layer.weight [n][i];
                            if (v.layer.weight [n][i] > Max) Max = v.layer.weight [n][i];
                        }
                        os << endl;
                    }
                    os << "   //    --- biases ---\n         ";
                    for (size_t n = 0; n < neuronCount; n++) {
                        os << v.layer.bias [n] << "f, ";
                        if (v.layer.bias [n] < Min) Min = v.layer.bias [n];
                        if (v.layer.bias [n] > Max) Max = v.layer.bias [n];
                    }
                    os << "\n   //    --- min: " << Min << ", max: " << Max << " ---";
                    int nextPowerOfTwo = 1;
                    while (nextPowerOfTwo < max (Max, -Min)) 
                        nextPowerOfTwo <<= 1;
                    os << " in case of quantization use at least Q<" << nextPowerOfTwo << ">\n";                    
                    os << v.layer.nextLayer.visualize ();
                    return os;
                }

                // initialize the model from C++ initializer list
                template<size_t N>
                neuralNetworkLayer_t& operator = (const float (&model) [N]) {
                    static_assert (N == sizeof (*this) / sizeof (float), "Wrong size of model!");
                    
                    memcpy ((void *) this, (void *) model, N * sizeof (float));
                    return *this;
                }

                // initialize the model from another model
                neuralNetworkLayer_t& operator = (const neuralNetworkLayer_t& other) {
                    if (this != &other)
                        memcpy (this, &other, sizeof (*this));
                    return *this;
                }
        };


    // output layer
        template <size_t inputCount, size_t activationFunction, size_t neuronCount> 
        class neuralNetworkLayer_t<inputCount, activationFunction, neuronCount> {

                // data structures needed for this layer: weight and bias
                float weight [neuronCount][inputCount];
                float bias [neuronCount];
                
            public:

                static constexpr size_t outputCount = neuronCount;
                using output_t = array<float, outputCount>;

                // calculates the output neurons of the neural network and returns the category that the input belongs to
                template<typename input_t>
                output_t forwardPass (const input_t (&input) [inputCount]) const {   
                    output_t neuron {};

                    // neuron = af (w x input + bias)
                        for (long n = 0; n < neuronCount; n++) {
                            neuron [n] = bias [n];
                            for (long i = 0; i < inputCount; i++)
                                neuron [n] += weight [n][i] * input [i];
                            neuron [n] = af<activationFunction> (neuron [n]);
                            // cout << "   output layer neuron [" << n << "] = " << neuron [n] << endl;
                        }

                    // start returning the result through all the previous layers
                        return neuron;
                }        

                // make it possible to use arrays instead of C arrays
                template<typename input_t>
                __attribute__((always_inline))
                inline output_t forwardPass (const array<input_t, inputCount> input) const {   
                    return forwardPass (*reinterpret_cast<const input_t (*)[inputCount]> (input.data ()));
                }


            // training

                // layer_t constructor - initialization of weight and bias
                neuralNetworkLayer_t () {
                    randomizeLayer ();
                }     
                
                void randomizeLayer () {
                    for (long n = 0; n < neuronCount; n++) {
                        for (long i = 0; i < inputCount; i++)
                            // weight are typically initialized with random numbers, He function is is particularly suitable to be used with ReLU activation function
                            weight [n][i] = randomInitializer<activationFunction> (inputCount, neuronCount);
                        // bias are typically set to 0 at initialization
                        bias [n] = 0;
                    }
                }
                
                void randomize () {
                    randomizeLayer ();
                }


                // update weight and bias in the output layer, returns the error
                template<typename input_t, typename expected_t>
                float backwardPropagation (const input_t (&input) [inputCount], const expected_t (&expected) [outputCount], float previousLayerDelta [inputCount] = NULL) { // the size of expected in all layers equals the size of the output of the output layer

                    // while moving forward do exactly the same as forwardPass function does
                        float z [neuronCount];
                        float neuron [neuronCount];

                        // z = weight x input + bias
                        // neuron = af (z)
                        for (long n = 0; n < neuronCount; n++) {
                            z [n] = bias [n];
                            for (long i = 0; i < inputCount; i++)
                                z [n] += weight [n][i] * input [i];
                            neuron [n] = af<activationFunction> (z [n]);
                        }

                    // calculate the error
                    float error = 0;
                    for (long n = 0; n < neuronCount; n++)
                        error += (neuron [n] - expected [n]) * (neuron [n] - expected [n]);
                    error = sqrt (error) / 2; 
                    // cout << "   output layer error = " << error << endl;

                    // update weight and bias at output layer
                        // delta = (neuron - expected) * a' (z)
                        // weight -= learningRate * delta * input
                        // bias -= learningRate * delta

                        float delta [neuronCount]; 
                        for (long n = 0; n < neuronCount; n++) {
                            // calculat delta at output layer    
                            delta [n] = (neuron [n] - expected [n]) * afDerivative<activationFunction> (z [n]);
                            // cout << "   output layer af' [" << n << "] = " << afDerivative<activationFunction> (z [n]) << endl;
                            // cout << "   output layer delta [" << n << "] = " << delta [n] << endl;

                            // update weight
                            for (long i = 0; i < inputCount; i++)
                                weight [n][i] -= learningRate * delta [n] * input [i];

                            // update bias
                            bias [n] -= learningRate * delta [n];
                        }

                    // calculate only the first part of previous layer delta, since z from previous layer is not available at this layer
                        // previousLayerDelta = weight * delta * a' (previous layer z)
                        if (previousLayerDelta) {
                            for (long i = 0; i < inputCount; i++) {
                                previousLayerDelta [i] = 0;
                                for (long n = 0; n < neuronCount; n++)
                                    previousLayerDelta [i] += weight [n][i] * delta [n];
                            }
                        }

                    return error;
                }

                // make it possible to use arrays instead of C arrays
                template<typename input_t, typename expected_t>
                __attribute__((always_inline))
                inline float backwardPropagation (const array<input_t, inputCount> input, const expected_t (&expected) [outputCount], float previousLayerDelta [inputCount] = NULL) {
                    return backwardPropagation (*reinterpret_cast<const input_t (*)[inputCount]> (input.data ()), expected, previousLayerDelta);
                }

                template<typename input_t, typename expected_t>
                __attribute__((always_inline))
                inline float backwardPropagation (const input_t (&input) [inputCount], const array<expected_t, outputCount> expected, float previousLayerDelta [inputCount] = NULL) {
                    return backwardPropagation (input, *reinterpret_cast<const expected_t (*)[outputCount]> (expected.data ()), previousLayerDelta);
                }

                template<typename input_t, typename expected_t>
                __attribute__((always_inline))
                inline float backwardPropagation (const array<input_t, inputCount> input, const array<expected_t, outputCount> expected, float previousLayerDelta [inputCount] = NULL) {
                    return backwardPropagation (*reinterpret_cast<const input_t (*)[inputCount]> (input.data ()), *reinterpret_cast<const expected_t (*)[outputCount]> (expected.data ()), previousLayerDelta);
                }


                // export the whole model as C++ initializer list
                friend ostream& operator << (ostream& os, const neuralNetworkLayer_t& nn) {
                    float *p = (float *) &nn;
                    for (long i = 0; i < sizeof (nn) / sizeof (float); i++) {
                        if (i > 0) {
                            os << ",";
                            if (i % 10 == 0)
                                os << endl;
                        }
                        os << *(p + i) << 'f';
                    }
                    return os;
                }

                // export visualized model as C++ initializer list
                struct visualized_t { const neuralNetworkLayer_t& layer; };
                visualized_t visualize () const { return { *this }; }
                friend ostream& operator << (ostream& os, const visualized_t& v) {
                    float Min = FLT_MAX;
                    float Max = -FLT_MAX;

                    os << "   // ----- layer inputs: " << inputCount << ", outputs (neurons): " << neuronCount << ", activation: " << afName<activationFunction> () << " -----\n";
                    os << "   //    --- weights ---\n";
                    for (long n = 0; n < neuronCount; n++) {
                        os << "         ";
                        for (long i = 0; i < inputCount; i++) {
                            os << v.layer.weight [n][i] << "f, ";
                            if (v.layer.weight [n][i] < Min) Min = v.layer.weight [n][i];
                            if (v.layer.weight [n][i] > Max) Max = v.layer.weight [n][i];
                        }
                        os << endl;
                    }
                    os << "   //    --- biases ---\n         ";
                    for (long n = 0; n < neuronCount; n++) {
                        os << v.layer.bias [n] << "f, ";
                        if (v.layer.bias [n] < Min) Min = v.layer.bias [n];
                        if (v.layer.bias [n] > Max) Max = v.layer.bias [n];
                    }
                    os << "\n   //    --- min: " << Min << ", max: " << Max << " ---";
                    int nextPowerOfTwo = 1;
                    while (nextPowerOfTwo < max (Max, -Min)) 
                        nextPowerOfTwo <<= 1;
                    os << " in case of quantization use at least Q<" << nextPowerOfTwo << ">\n";                    
                    return os;
                }

                // initialize the model from C++ initializer list
                template<long N>
                neuralNetworkLayer_t& operator = (const float (&model) [N]) {
                    static_assert (N == sizeof (*this) / sizeof (float), "Wrong size of model!");
                    
                    memcpy ((void *) this, (void *) model, N * sizeof (float));
                    return *this;
                }

                // initialize the model from another model
                neuralNetworkLayer_t& operator = (const neuralNetworkLayer_t& other) {
                    if (this != &other)
                        memcpy (this, &other, sizeof (*this));
                    return *this;
                }
        };


    // ----- softmax -----

    template<size_t N, typename T>
    array<T, N> softmax (const array<T, N>& input) {
        array<T, N> output {};
    
        // softmax normalization
        T sum = 0;
        for (long n = 0; n < N; n++)
            sum += exp (input [n]);
        for (long n = 0; n < N; n++)
            if (sum > 0)
                output [n] = exp (input [n]) / sum;
            else
                output [n] = 0;
    
        return output;
    }

#endif
