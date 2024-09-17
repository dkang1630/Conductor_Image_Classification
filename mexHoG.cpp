//MATLAB API Header Files
#include "mex.hpp"
#include "mexAdapter.hpp"

//Header for HoG function
#include "HoG.cpp"

using namespace matlab::data;
using matlab::mex::ArgumentList;


class MexFunction : public matlab::mex::Function {
public:
    void operator() (ArgumentList outputs, ArgumentList inputs) {
    //Validate arguments
    checkArguments(outputs, inputs);

    //Implement function of interest
    //Parse inputs
    matlab::data::TypedArray<double> doubleArray = std::move(inputs[0]); //to define specific types for the inputs. Double in this case
    //Run function on each element of input array
    for (auto& i : doubleArray) {
        
       double pixels[] = {elem}; //pixels in an array
       double params[] = {}
       
       HoG(pixels, params, )
    }
    //Assign oututs
    outputs[0] = doubleArray;
    }
};

void checkArguments(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
    //Create pointer to MATLAB engine;
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
    //Create array factor, allows us to create MATALB arrays in C++
    matlab::data::ArrayFactory factory;
    //Check input sizes and types
    if (inputs[0].getType() != ArrayType::DOUBLE ||
        inputs[0].getType() == ArrayType::COMPLEX_DOUBLE)
    {
        //Throw error directly into MATLAB if type does not match
        matlabPtr->feval(u"error", 0,
            std::vector<Array>({ factory.createScalar("Input must double array")}));
    }
    //Check output size
    if (outputs.size() > 1) {
        matlabPtr ->feval(u"error", 0,
            std::vector<Array>({ factory.createScalar("Only one output is returned")}));

    }
}