/*

    compatibility.h

    This file is part of Clap Recognition Using a Neural Network from Scratch (C++ for Arduino): https://github.com/BojanJurca/Neural-Network-from-Scratch-for-Arduino-A-Clap-Recognition-System

    The purpose of this library is to provide compatibility between Arduino and standard C++, allowing sketches 
    to be compiled on more powerful machines. This enables time-intensive tasks, such as neural network training, 
    to be performed externally, while only the trained model is imported back to Arduino.

    The library offers essential standard C++ functionalities, such as cout and support for complex numbers, bringing 
    them to the Arduino environment.

    Additionally, it provides certain Arduino-specific functionalities—such as String support—to standard C++, enabling 
    sketches to be compiled with a regular C++ compiler. However, hardware-dependent functions like pinMode are included 
    for compatibility but do not function in a standard C++ environment.

    Bojan Jurca, Oct 10, 2025

*/


#ifndef __COMPATIBILITY_H__
    #define __COMPATIBILITY_H__


    #ifdef ARDUINO 

        // ----- essential standard C++ functionalities for Arduino -----


            // standard C++ rand and srand for Arduino instead of random

                #define rand() random(RAND_MAX)
                #define srand(X) randomSeed(X)
                #ifdef ARDUINO_ARCH_AVR // Assuming Arduino Mega or Uno
                    // introduce time function only for the purpose rand (time (NULL)) would work
                    unsigned long time (void *p) { return millis (); }
                #endif


        // cout instead of Arduino Serial

            // cinit instead of Arduino Serial.begin
            #ifdef ARDUINO_ARCH_AVR // Assuming Arduino Mega or Uno
                void cinit (bool waitForSerial = false, unsigned int serialSpeed = 9600, unsigned int waitAfterSerial = 1000) {
                    Serial.begin (serialSpeed);
                    if (waitForSerial)
                        while (!Serial) 
                            delay (10);
                    delay (waitAfterSerial);
                }
            #else
                void cinit (bool waitForSerial = false, unsigned int serialSpeed = 115200, unsigned int waitAfterSerial = 1000) {
                    Serial.begin (serialSpeed);
                    if (waitForSerial)
                        while (!Serial) 
                            delay (10);
                    delay (waitAfterSerial);
                }
            #endif

            #define endl "\r\n"

            // cout instead of Arduino Serial.print ...
            class ostream {
                public:

                    template<typename T>
                    ostream& operator << (const T& value) {
                        Serial.print (value);            
                        return *this;
                    }
            };

            // Create a working instances
            ostream cout;


        // arrays
            template<class T, size_t N>
            class array {
                private:
                    T _a_ [N];

                public:
                    inline T &operator [] (size_t ind) __attribute__((always_inline)) { 
                        return _a_ [ind]; 
                    }

                    class iterator {
                        public:
                            iterator (T *p) { _p_ = p; }
                            T& operator *() const { return *_p_; }
                            iterator& operator ++ () { ++ _p_; return *this; }
                            // C++ will stop iterating when != operator returns false, this is when __position__ counts to vector.size ()
                            friend bool operator != (const iterator& a, const iterator& b) { return a._p_ != b._p_; }
                            friend bool operator == (const iterator& a, const iterator& b) { return a._p_ == b._p_; }

                        private:
                            T *_p_;

                    };

                    iterator begin () { return iterator (_a_); }    // first element
                    iterator end () { return iterator (_a_ + N); }  // past the last element
            };


        // complex numbers for Arduino - only these member functions that are needed
            template<class T>
            class complex {
                private:
                    T _real_;
                    T _imag_;
                
                public:
                    // Constructor to initialize real and imag to 0
                    complex() : _real_ (0), _imag_ (0) {}
                    complex(T real, T imag) : _real_ (real), _imag_ (imag) {}
                
                    // Real and imag parts
                    inline T real () const __attribute__((always_inline)) { return _real_; }
                    inline T imag () const __attribute__((always_inline)) { return _imag_; }
                
                    // + operator
                    template<typename t>
                    friend complex operator + (const complex<T>& obj1, const complex<t>& obj2) {
                        return {obj1.real () + obj2.real (), obj1.imag () + obj2.imag ()};
                    }
                
                    // += operator
                    complex& operator += (const complex<T>& other) {
                        _real_ += other.real ();
                        _imag_ += other.imag ();
                        return *this;
                    }
                
                    // - operator
                    template<typename t>
                    friend complex operator - (const complex<T>& obj1, const complex<t>& obj2) {
                        return {obj1.real () - obj2.real (), obj1.imag () - obj2.imag ()};
                    }
                
                    // -= operator
                    complex& operator -= (const complex<T>& other) {
                        _real_ -= other.real ();
                        _imag_ -= other.imag ();
                        return *this;
                    }
                
                    // * operator
                    template<typename t>
                    friend complex operator * (const complex<T>& obj1, const complex<t>& obj2) {
                        return {obj1.real () * obj2.real () - obj1.imag () * obj2.imag (), obj1.imag () * obj2.real () + obj1.real () * obj2.imag ()};
                    }
                
                    // *= operator
                    complex& operator *= (const complex<T>& other) {
                        T r = real () * other.real () - imag () * other.imag ();
                        T i = imag () * other.real () + real () * other.imag ();
                        _real_ = r;
                        _imag_ = i;
                        return *this;
                    }
                
                    // / operator
                    template<typename t>
                    friend complex operator / (const complex<T>& obj1, const complex<t>& obj2) {
                        T tmp = obj2.real () * obj2.real () + obj2.imag () * obj2.imag ();
                        return {(obj1.real () * obj2.real () + obj1.imag () * obj2.imag ()) / tmp, (obj1.imag () * obj2.real () - obj1.real () * obj2.imag ()) / tmp};
                    }
                
                    // /= operator
                    complex& operator /= (const complex<T>& other) {
                        T tmp = other.real () * other.real () + other.imag () * other.imag ();
                        T r = (real () * other.real () + imag () * other.imag ()) / tmp;
                        T i = (imag () * other.real () - real () * other.imag ()) / tmp;
                        _real_ = r;
                        _imag_ = i;
                        return *this;
                    }
                
                    // conjugate function
                    constexpr complex conj () const { return {real (), -imag ()}; }
                
                    // print complex number to ostream
                    friend ostream& operator << (ostream& os, const complex& c) {
                        os << c.real () << '+' << c.imag () << 'i';
                        return os;
                    }
            };

            complex<float> exp (complex<float> z) {
                float exp_real = expf (z.real ());
                return { exp_real * cos (z.imag ()), exp_real * sin (z.imag ()) };
            }

            complex<double> exp (complex<double> z) {
                double exp_real = exp (z.real ());
                return { exp_real * cos (z.imag ()), exp_real * sin (z.imag ()) };
            }

            // replace abs #definition with function template that would also handle complex numbers
            #ifdef abs
                #undef abs

                template<typename T>
                T abs (T x) { return x > 0 ? x : -x; }

                float abs (const complex<float>& z) { return sqrt (z.real () * z.real () + z.imag () * z.imag ()); }

                double abs (const complex<double>& z) { return sqrt (z.real () * z.real () + z.imag () * z.imag ()); }

            #endif
    

    #else 

        // ----- Arduino-specific functionalities—such as String support—to standard C++ -----

            using namespace std;
            #include <cmath>
            #include <iostream>
            #include <bits/stdc++.h>
            #include <array>


        // call Arduino setup () and loop () from standard C++ main ()

            extern void setup ();
            extern void loop ();

            int main () {
                setup ();

                while (true)
                    loop ();
                return 0;
            }        
    

        // some definitions are there only for the code to compile

            // cinit
            void cinit (bool waitForSerial = false, unsigned int serialSpeed = 9600, unsigned int waitAfterSerial = 1000) {}

            // pinMode
            #define INPUT 1
            #define OUTPUT 2
            void pinMode (int pin, int mode) {};
        
            // simulate Arduino digitalRead and digitalWrite
            unsigned char digitalRead (int pin) { return 0; }
            void digitalWrite (int pin, int value) { cout << "pin [" << pin << "] = " << value << endl; }

            // AD registers
            unsigned char ADCSRA;
            #define ADPS0 0
            #define ADPS1 1
            #define ADPS2 2
            #define ADSC 3
            #define ADC 4
            #define ADEN 5
            #define REFS0 6
            unsigned char ADMUX;
            unsigned char bit (unsigned char b) { return b; }
            void bitSet (unsigned char a, unsigned char b) {}
            bool bit_is_clear (unsigned char a, unsigned char b) { return true; }

            // delay
            void delay (unsigned long milliseconds) {}

    #endif

#endif
