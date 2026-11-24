#include <iostream>
#include <sstream>
#include <cstring>
#include <vector>
#include <string>
#include <malloc.h>
#include <chrono>
#include <iomanip>

#include "treeNode.hpp"
#include "funcDef.hpp"


// Global memory tracking variables
long long g_current_bytes = 0;
long long g_stage_peak_bytes = 0;

// Overload the new function to record memory it allocates every call
void* operator new(std::size_t size) {
    void* ptr = std::malloc (size);
    if (ptr != nullptr)
    {
        g_current_bytes += _msize (ptr);
        g_stage_peak_bytes = std::max(g_stage_peak_bytes, g_current_bytes);
    }
    return ptr;
}

// Overload the delete function to record memory it frees
void operator delete(void* ptr, std::size_t size) noexcept {
    if (g_current_bytes >= size && ptr != nullptr)
        g_current_bytes -= _msize (ptr);
    std::free(ptr);
}

void operator delete(void* ptr) noexcept {
    if (ptr != nullptr)
        g_current_bytes -= _msize (ptr);
    std::free(ptr);
}

// Function declaration to measure time duration for each stage
template <typename Func>
void measureStage(const std::string& stageName, Func func) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto start_mem = g_current_bytes;
    g_stage_peak_bytes = g_current_bytes;
    func();
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
    double peak_kb = static_cast<double>(g_stage_peak_bytes - start_mem) / 1024.0;
    double net_kb  = static_cast<double>(g_current_bytes - start_mem) / 1024.0;
    std::cout << std::left  << std::setw(20) << stageName 
              << " | Time: " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << duration_ms << " ms"
              << " | Peak: " << std::setw(9) << peak_kb << " KB"
              << " | Net: "  << std::setw(9) << net_kb << " KB\n";
}



int main (int argc, char *argv[])
{
    splitting criterion;
    dataSets inputData;
    tree *classifier = nullptr;
    if (argc != 3 && argc != 5)
    {
        std::cout << "Wrong format used: ./main <Splitting Criteria> <Data Set>\n"
                  << "                   ./main <Splitting Criteria> <Data Set> <Input Training File> <Input Validation File>" << std::endl;
        return 1;
    }

    // Check to see which classification the users wants to use
    if (std::strcmp (argv[1], "GAIN") == 0)
        criterion = GAIN;
    else if (std::strcmp (argv[1], "GAIN_RATIO") == 0)
        criterion = GAIN_RATIO;
    else if (std::strcmp (argv[1], "GINI") == 0)
        criterion = GINI;
    else
    {
        std::cout << "Incorrect splitting criteria provided, please use either:\n"
                    << "GAIN, GAIN_RATIO, GINI" << std::endl;
        return 0;
    }

    // Check to see which data set the user selected
    if (std::strcmp (argv[2], "iris") == 0)
        inputData = iris;
    else if (std::strcmp (argv[2], "abalone") == 0)
        inputData = abalone;
    else if (std::strcmp (argv[2], "car") == 0)
        inputData = car;
    else if (std::strcmp (argv[2], "connect-4") == 0)
        inputData = connect4;
    else if (std::strcmp (argv[2], "chess") == 0)
        inputData = chess;
    else if (std::strcmp (argv[2], "tictactoe") == 0)
        inputData = tictactoe;
    else if (std::strcmp (argv[2], "nursery") == 0)
        inputData = nursery;
    else if (std::strcmp (argv[2], "gamma") == 0)
        inputData = gamma;
    else if (std::strcmp (argv[2], "spam") == 0)
        inputData = spam;
    else if (std::strcmp (argv[2], "poker") == 0)
        inputData = poker;
    else if (std::strcmp (argv[2], "tennis") == 0)
        inputData = tennis;
    else
    {
        std::cout << "Incorrect data set name provided, please use either:\n"
                    << "iris\nabalone\ncar\nconnect-4\nchess\ntictactoe\nnursery\ngamma\nspam\npoker\ntennis" << std::endl;
        return 0;
    }

    // Read data from input file into 2d vector
    std::vector<std::vector<std::string>> Data;
    std::vector<std::string> featureName;
    std::vector<bool> featureCont;
    bool automated = false;
    if (argc == 5)
        automated = true;
    switch (inputData)
    {
        case iris: readData ((automated) ? argv[3] : "iris.data", Data); featureName = irisName; featureCont = irisCont; break;
        case abalone: readData ((automated) ? argv[3] : "abalone.data", Data); featureName = abaloneName; featureCont = abaloneCont; break;
        case car: readData ((automated) ? argv[3] : "car.data", Data); featureName = carName; featureCont = carCont; break;
        case connect4: readData ((automated) ? argv[3] : "connect-4.data", Data); featureName = connect4Name; featureCont = connect4Cont; break;
        case chess: readData ((automated) ? argv[3] : "kr-vs-kp.data", Data); featureName = chessName; featureCont = chessCont; break;
        case tictactoe: readData ((automated) ? argv[3] : "tic-tac-toe.data", Data); featureName = tictactoeName; featureCont = tictactoeCont; break;
        case nursery: readData ((automated) ? argv[3] : "nursery.data", Data); featureName = nurseryName; featureCont = nurseryCont; break;
        case gamma: readData ((automated) ? argv[3] : "magic04.data", Data); featureName = gammaName; featureCont = gammaCont; break;
        case spam: readData ((automated) ? argv[3] : "spambase.data", Data); featureName = spamName; featureCont = spamCont; break;
        case poker: readData ((automated) ? argv[3] : "poker-hand-testing.data", Data); featureName = pokerName; featureCont = pokerCont; break;
        case tennis: readData ((automated) ? argv[3] : "tennis.data", Data); featureName = tennisName; featureCont = tennisCont; break;
    }

    // Separate either running the automation script with measure the performance and accuracy or user mode
    if (!automated)      // User Mode
    {
         // Train/build classification tree
        classifier = new tree (featureName, featureCont, criterion);
        classifier->train (Data);

        // Take user inputs and feed into classifier and output back what the result is
        std::cout << "Input values to classify separated by spaces:" << std::endl;

        std::vector<std::string> sample;
        std::string line, token;

        while (1)
        {
            sample.clear ();
            std::cout << "Values: ";
            std::getline (std::cin, line);
            if (line == "exit" || line == "Exit")
                break;
            std::stringstream ss (line);
            while (ss >> token)
                sample.push_back (token);
            if (sample.size () != Data[0].size () - 1)
            {
                std::cout << "Incorrect number of parameters were entered, expected " << Data[0].size () - 1
                << " parameters but received " << sample.size () << " parameters" << std::endl;
                continue;
            }
            std::cout << "Predicted: " << classifier->predict (sample) << std::endl;
        }
    } else      // Automated Mode
    {
        // Read the validation data for prediction use.
        std::vector<std::vector<std::string>> validationData;
        size_t totalPredicts = 0, correctPredicts = 0, nullPredicts = 0;
        readData (argv[4], validationData);

        measureStage ("Train", [&]()
        {
            // Train/build classification tree
            classifier = new tree (featureName, featureCont, criterion);
            classifier->train (Data);
        });

        measureStage ("Predict", [&]()
        {
            for (const auto &row : validationData)
            {
                std::vector<std::string> input (row.begin (), row.end () - 1);
                std::string result = classifier->predict (input);
                if (result == "NULL")
                    nullPredicts++;
                if (result == row.back ())
                    correctPredicts++;
                totalPredicts++;
            }
        });
        std::cout << "Accuracy: " << (float)correctPredicts / totalPredicts * 100 << " | NULL Count: " << nullPredicts << std::endl;
    }

    delete classifier;
    return 0;
}