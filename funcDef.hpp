#ifndef FUNC_DEF
#define FUNC_DEF

#include <unordered_map>
#include <vector>
#include <string>
#include <unordered_set>

// Create enum class for splitting criterion
enum splitting {
    GAIN,
    GAIN_RATIO,
    GINI
};


// Create enum class for different data sets
enum dataSets {
    iris,
    abalone,
    car,
    connect4,
    chess,
    tictactoe,
    nursery,
    gamma,
    spam,
    poker,
    tennis
};

// Create vector to define which columns in data are discrete or continuous
inline static std::vector<bool> irisCont = {true, true, true, true};
inline static std::vector<bool> abaloneCont = {false, true, true, true, true, true, true, true};
inline static std::vector<bool> carCont(6, false);
inline static std::vector<bool> connect4Cont(42, false);
inline static std::vector<bool> chessCont(36, false);
inline static std::vector<bool> tictactoeCont(9, false);
inline static std::vector<bool> nurseryCont(8, false);
inline static std::vector<bool> gammaCont(10, true);
inline static std::vector<bool> spamCont(57, true);
inline static std::vector<bool> pokerCont (10, false);
inline static std::vector<bool> tennisCont (4, false);

// Create vector to define which columns get which feature name
inline static std::vector<std::string> irisName = {"sepalLength", "sepalWidth", "petalLength", "petalWidth"};
inline static std::vector<std::string> abaloneName = {"Sex", "Length", "Diameter", "Height", "WholeWeight", "ShuckedWeight", "VisceraWeight", "ShellWeight"};
inline static std::vector<std::string> carName = {"buying", "maint", "doors", "persons", "lug_boot", "safety"};
inline static std::vector<std::string> connect4Name = {"a1", "a2", "a3", "a4", "a5", "a6", "b1", "b2", "b3", "b4", "b5", "b6",
                                                        "c1", "c2", "c3", "c4", "c5", "c6", "d1", "d2", "d3", "d4", "d5", "d6", 
                                                        "e1", "e2", "e3", "e4", "e5", "e6", "f1", "f2", "f3", "f4", "f5", "f6", 
                                                        "g1", "g2", "g3", "g4", "g5", "g6"};
inline static std::vector<std::string> chessName = {"bkblk", "bknwy", "bkon8", "bkona", "bkspr", "bkxbq", "bkxcr", "bkxwp", "blxwp", "bxqsq", "cntxt", "dsopp", "dwipd", 
                                                    "hdchk", "katri", "mulch", "qxmsq", "r2ar8", "reskd", "reskr", "rimmx", "rkxwp", "rxmsq", "simpl", "skach", "skewr", "skrxp",
                                                    "spcop", "stlmt", "thrsk", "wkcti", "wkna8", "wknck", "wkovl", "wkpos", "wtoeg"};
inline static std::vector<std::string> tictactoeName = {"topLeft", "topMid", "topRight", "midLeft", "midMid", "midRight", "botLeft", "botMid", "botRight"};
inline static std::vector<std::string> nurseryName = {"parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"};
inline static std::vector<std::string> gammaName = {"fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist"};
inline static std::vector<std::string> spamName = {
    "f1",  "f2",  "f3",  "f4",  "f5",  "f6",  "f7",  "f8",  "f9",  "f10",
    "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20",
    "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "f30",
    "f31", "f32", "f33", "f34", "f35", "f36", "f37", "f38", "f39", "f40",
    "f41", "f42", "f43", "f44", "f45", "f46", "f47", "f48", "f49", "f50",
    "f51", "f52", "f53", "f54", "f55", "f56", "f57"
};
inline static std::vector<std::string> pokerName = {
    "S1", "C1", "S2", "C2", "S3", "C3", "S4", "C4", "S5", "C5"
};
inline static std::vector<std::string> tennisName = {"Outlook", "Temperature", "Humidity", "Wind"};


int inspectData (const std::string inputFile, std::vector<std::vector<std::string>> &Data);
int readData (const std::string inputFile, std::vector<std::vector<std::string>> &Data);
int countClassTypes (const std::vector<std::vector<std::string>> &Data, std::unordered_map<std::string, size_t> &classCounts);
float calcEntropy (const std::unordered_map<std::string, size_t> &classCounts, const size_t rows);
float evalContGain (const std::vector<std::vector<std::string>> &Data, const size_t col, const float parEntropy, float &threshold);
float evalDiscGain (const std::vector<std::vector<std::string>> &Data, const size_t col, const float parEntropy);
std::string findMajorityLabel (const std::unordered_map<std::string, size_t> &classCounts);
float splitInfo (const std::vector<size_t> &branchSizes, const size_t rows);
float evalContGainRatio (const std::vector<std::vector<std::string>> &Data, const size_t col, const float parEntropy, float &threshold);
float evalDiscGainRatio (const std::vector<std::vector<std::string>> &Data, const size_t col, const float parEntropy);
std::vector<size_t> calcBranchSizes (const std::vector<std::vector<std::string>> &Data, const size_t col);
float calcGini (const std::unordered_map<std::string, size_t> &classCounts, const size_t rows);
float evalContGini (const std::vector<std::vector<std::string>> &Data, const size_t col, const float parGini, float &threshold);
float evalDiscGini (const std::vector<std::vector<std::string>> &Data, const size_t col, const float parGini, std::unordered_set<std::string> &leftSet);
float calcBestGain (const bool cont, const float parentG, const splitting criterion, const size_t col, float &threshold, 
    std::unordered_set<std::string> &leftThreshold, const std::vector<std::vector<std::string>> &Data);

#endif