/*

    neuralNetwork.hpp

    This file is part of Clap Recognition Using a Neural Network from Scratch (C++ for Arduino): https://github.com/BojanJurca/Clap-Recognition-Using-a-Neural-Network-from-Scratch-Cpp-for-Arduino

    The neural network is implemented as a C++ variadic template. This approach was chosen to simplify both the usage 
    of the library and the modification of the network topology, making it as intuitive and flexible as possible.


        //                   .--- the number of neurons in the first layer - it corresponds to the size of the patterns that neural network will use to make the categorization
        //                   |    .--- second layer activation function
        //                   |    |    .--- the number of neurons the second layer 
        //                   |    |    |                               .--- output layer activation function
        //                   |    |    |                               |      .--- the number of neurons in the output layer - it corresponds to the number of categories that the neural network recognizes
        //                   |    |    |                               |      |
        neuralNetworkLayer_t<13, ReLU, 16, / * add more if needed * / Sigmoid, 2> neuralNetwork;

    
    Bojan Jurca, May 22, 2025

*/


#include "compatibility.h" // Arduino <-> standard C++ compatibility 


#ifndef __NEURAL_NETWORK_HPP__
    #define __NEURAL_NETWORK_HPP__


    // non-linear neuron activation function and its derivative

        static float __Sigmoid__ (float x) { return 1.f / (1.f + expf (-x)); }
        static float __SigmoidDerivative__ (float x) { float s = __Sigmoid__ (x); return s * (1 - s); }
        
        static float __ReLU__ (float x) { return x <= 0.f ? 0.f : x; }
        static float __ReLUDerivative__ (float x) { return x <= 0.f ? 0.f : 1.f; }

        #define Sigmoid 0
        #define ReLU 1

        float (*__af__ []) (float) = {__Sigmoid__, __ReLU__};
        float (*__af_derivative__ []) (float) = {__SigmoidDerivative__, __ReLUDerivative__};


    // training


        // initialization function Xavier returns normally distributed random number with a mean of 0 and a standard deviation of sqrt (2/ (inputCount + neuronCount)
        // it is usually used with Sigmoid activation function
        float __Xavier__ (size_t inputCount, size_t neuronCount) {
            // let's use Box-Muller Transform to calculate normaly distributed random variable from uniformly distrubuted random function

                // 1. select 2 independent uniformly distributed random values in interval (0, 1)
                float U1 = ((float) (rand () % (RAND_MAX - 1) + 1)) / RAND_MAX;
                float U2 = ((float) (rand () % (RAND_MAX - 1) + 1)) / RAND_MAX;
                
                // 2. use Box-Muller transtor to tranform them to two independed normally distributed random values with mean of 0 and variance of 1
                // float N1 = sqrt (-2 * log (U1)) * cos (2 * M_PI * U2); 
                float N2 = sqrt (-2 * log (U1)) * sin (2 * M_PI * U2); // we don't actually need the second independent random variable here

            // apply the desired mean of 0 and variance of sqrt (2.0 / (inputCount + neuronCount)) to random variable N1
                return 0 + sqrt (2.0 / (inputCount + neuronCount)) * N2; 
        }
        
        // initialization function He returns normally distributed random number with a mean of 0 and a standard deviation of sqrt (2/inputCountn)
        // it is usually used with ReLU activation function
        float __He__ (size_t inputCount, size_t neuronCount) {
            // let's use Box-Muller Transform to calculate normaly distributed random variable from uniformly distrubuted random function

                // 1. select 2 independent uniformly distributed random values in interval (0, 1)
                float U1 = ((float) (rand () % (RAND_MAX - 1) + 1)) / RAND_MAX;
                float U2 = ((float) (rand () % (RAND_MAX - 1) + 1)) / RAND_MAX;
                
                // 2. use Box-Muller transtor to tranform them to two independed normally distributed random values with mean of 0 and variance of 1
                float N1 = sqrt (-2 * log (U1)) * cos (2 * M_PI * U2); 
                // float N2 = sqrt (-2 * log (U1)) * sin (2 * M_PI * U2); // we don't actually need the second independent random variable here

            // apply the desired mean of 0 and variance of sqrt (2.0 / inputCount) to random variable N1
                return 0 + sqrt (2.0 / inputCount) * N1; 
        }
        
        float (*__randomInitializer__ []) (size_t, size_t) = {__Xavier__, __He__};        
        
        // learning rate
        #define learningRate 0.01


    // basic neuralNetwork_t class template, not used but needed by C++ compiler
        template <size_t... sizes> 
        class neuralNetworkLayer_t;


    // hidden layers
        template <size_t inputCount, size_t af, size_t neuronCount, size_t... sizes> 
        class neuralNetworkLayer_t<inputCount, af, neuronCount, sizes...> {

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
                output_t forwardPass (const input_t (&input) [inputCount]) {   
                    float neuron [neuronCount];

                    // neuron = a (w x input + bias)
                        for (size_t n = 0; n < neuronCount; n++) {
                            neuron [n] = bias [n];
                            for (size_t i = 0; i < inputCount; i++)
                                neuron [n] += weight [n][i] * input [i];
                            neuron [n] = __af__ [af] (neuron [n]);
                        }

                    // return what the next layer thinks about the neurons clculated here
                        return nextLayer.forwardPass (neuron);
                }
            

            // training

                // layer_t constructor - initialization of weight and bias
                neuralNetworkLayer_t () {
                    for (size_t n = 0; n < neuronCount; n++) {
                        for (size_t i = 0; i < inputCount; i++)
                            // weight are typically initialized with random numbers, He function is is particularly suitable to be used with ReLU activation function
                            weight [n][i] = __randomInitializer__ [af] (inputCount, neuronCount);
                        // bias are typically set to 0 at initialization
                        bias [n] = 0;
                    }
                }


                // iterate from the last layer to the first and adjust weight and bias meanwhile, returns the error clculated at output layer
                template<typename input_t, typename expected_t>
                float backwardPropagation (const input_t (&input) [inputCount], const expected_t (&expected) [outputCount], float previousLayerDelta [inputCount] = NULL) { // the size of expected in all layers equals the size of the output of the output layer

                    // while moving forward do exactly the same as forwardPass function does
                        float z [neuronCount];
                        float neuron [neuronCount];

                        // z = weight x input + bias
                        // neuron = a (z)
                        for (size_t n = 0; n < neuronCount; n++) {
                            z [n] = bias [n];
                            for (size_t i = 0; i < inputCount; i++)
                                z [n] += weight [n][i] * input [i];
                            neuron [n] = __af__ [af] (z [n]);
                        }

                    // calculate the first part of delta in the next layer then apply activation function derivative here
                        // delta = next layer weight * next layer delta * a' (z)
                        float delta [neuronCount]; 
                        float error = nextLayer.backwardPropagation (neuron, expected, delta);
                        // calculate only the second part of delta, the first par has already been calculated at the next layer
                        for (size_t n = 0; n < neuronCount; n++)
                            delta [n] *= __af_derivative__ [af] (z [n]);

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

                // export the whole model as C++ initializer list
                friend ostream& operator << (ostream& os, const neuralNetworkLayer_t& nn) {
                    int32_t *p = (int32_t *) &nn;
                    os << "{";
                    for (size_t i = 0; i < sizeof (nn) / sizeof (int32_t); i++) {
                        if (i > 0) {
                            os << ",";
                            if (i % 16 == 0)
                                os << endl;
                        }
                        os << *(p + i);
                    }
                    os << "}\n";
                    return os;
                }
                
                template<size_t N>
                neuralNetworkLayer_t& operator = (const int32_t (&model) [N]) {
                    static_assert (N == sizeof (*this) / sizeof (int32_t), "Wrong size of model!");
                    memcpy ((void *) this, (void *) model, N * sizeof (int32_t));
                    return *this;
                }                

        };


    // output layer
        template <size_t inputCount, size_t af, size_t neuronCount> 
        class neuralNetworkLayer_t<inputCount, af, neuronCount> {

                // data structures needed for this layer: weight and bias
                float weight [neuronCount][inputCount];
                float bias [neuronCount];

            public:

                static constexpr size_t outputCount = neuronCount;
                using output_t = array<float, outputCount>;

                // calculates the output neurons of the neural network and returns the category that the input belongs to
                template<typename input_t>
                output_t forwardPass (const input_t (&input) [inputCount]) {   
                    output_t neuron {};

                    // neuron = a (w x input + bias)
                        for (size_t n = 0; n < neuronCount; n++) {
                            neuron [n] = bias [n];
                            for (size_t i = 0; i < inputCount; i++)
                                neuron [n] += weight [n][i] * input [i];
                            neuron [n] = __af__ [af] (neuron [n]);
                        }

                    // softmax normalization of the result
                        float sum = 0;
                        for (size_t n = 0; n < neuronCount; n++)
                            sum += expf (neuron [n]);
                        for (size_t n = 0; n < neuronCount; n++)
                            if (sum > 0)
                                neuron [n] = expf (neuron [n]) / sum;
                            else
                                neuron [n] = 0;

                    // start returning the result through all the previous layers
                        return neuron;
                }        


            // training

                // layer_t constructor - initialization of weight and bias
                neuralNetworkLayer_t () {
                    for (size_t n = 0; n < neuronCount; n++) {
                        for (size_t i = 0; i < inputCount; i++)
                            // weight are typically initialized with random numbers, He function is is particularly suitable to be used with ReLU activation function
                            weight [n][i] = __randomInitializer__ [af] (inputCount, neuronCount);
                        // bias are typically set to 0 at initialization
                        bias [n] = 0;
                    }
                }        


                // update weight and bias in the output layer, returns the error
                template<typename input_t, typename expected_t>
                float backwardPropagation (const input_t (&input) [inputCount], const expected_t (&expected) [neuronCount], float previousLayerDelta [inputCount] = NULL) { // the size of expected in all layers equals the size of the output of the output layer

                    // while moving forward do exactly the same as forwardPass function does
                        float z [neuronCount];
                        float neuron [neuronCount];

                        // z = weight x input + bias
                        // neuron = a (z)
                        for (size_t n = 0; n < neuronCount; n++) {
                            z [n] = bias [n];
                            for (size_t i = 0; i < inputCount; i++)
                                z [n] += weight [n][i] * input [i];
                            neuron [n] = __af__ [af] (z [n]);
                        }

                    // calculate the error
                    float error = 0;
                    for (size_t n = 0; n < neuronCount; n++)
                        error += (expected [n] - neuron [n]) * (expected [n] - neuron [n]);
                    error = sqrt (error) / 2; 

                    // update weight and bias at output layer
                        // delta = (neuron - expected) * a' (z)
                        // weight -= learningRate * delta * input
                        // bias -= learningRate * delta

                        float delta [neuronCount]; 
                        for (size_t n = 0; n < neuronCount; n++) {
                            // calculat delta at output layer    
                            delta [n] = (neuron [n] - expected [n]) * __af_derivative__ [af] (z [n]);

                            // update weight
                            for (size_t i = 0; i < inputCount; i++)
                                weight [n][i] -= learningRate * delta [n] * input [i];

                            // update bias
                            bias [n] -= learningRate * delta [n];
                        }

                    // calculate only the first part of previous layer delta, since z from previous layer is not available at this layer
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


                // export the whole model as C++ initializer list
                friend ostream& operator << (ostream& os, const neuralNetworkLayer_t& nn) {
                    int32_t *p = (int32_t *) &nn;
                    os << "{";
                    for (size_t i = 0; i < sizeof (nn) / sizeof (int32_t); i++) {
                        if (i > 0) {
                            os << ",";
                            if (i % 16 == 0)
                                os << endl;
                        }
                        os << *(p + i);
                    }
                    os << "}\n";
                    return os;
                }
                
                template<size_t N>
                neuralNetworkLayer_t& operator = (const int32_t (&model) [N]) {
                    static_assert (N == sizeof (*this) / sizeof (int32_t), "Wrong size of model!");
                    memcpy ((void *) this, (void *) model, N * sizeof (int32_t));
                    return *this;
                }
        };

#endif
