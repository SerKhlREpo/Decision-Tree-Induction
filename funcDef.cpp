#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <unordered_set>

#include "funcDef.hpp"

int inspectData (const std::string inputFile, std::vector<std::vector<std::string>> &Data)
{
    // Create variable to store the number of elements per row and number of rows
    size_t features = 0;
    size_t transacts = 0;
    // convert the testfile into a input file stream
    std::ifstream inputData (inputFile);
    if (!inputData.is_open ())
    {
        std::cout << "Unable to find " << inputFile << " in application folder." << std::endl;
        exit (1);
    }

    // Keep reading line by line
    std::string line;
    while (std::getline (inputData, line))
    {
        if (line.empty ())
            continue;
        // for the first row, note the number of features
        if (transacts < 1)
        {
            std::stringstream stream (line);
            std::string token;
            while (std::getline (stream, token, ','))
                features++;
        }
        // increment for the number of rows in the text file.
        ++transacts;
    }

    //Preallocate space in Data
    Data.reserve (transacts);
    std::vector<std::string> dataRow (features, "");
    for (size_t i = 0; i < transacts; i++)
        Data.push_back (dataRow);

    inputData.close ();
    return 0;
}


int readData (const std::string inputFile, std::vector<std::vector<std::string>> &Data)
{
    // convert the testfile into a input file stream
    std::ifstream inputData (inputFile);
    if (!inputData.is_open ())
    {
        std::cout << "Unable to find " << inputFile << " in application folder." << std::endl;
        exit (1);
    }

    // Preallocate space in Data vector
    inspectData (inputFile, Data);

    // Keep reading line by line
    std::string line;
    size_t lineIndex = 0;
    while (std::getline (inputData, line))
    {
        // Safe guard to protect if a line is empty but we didn't reach the end of the file
        if (line.empty ())
            continue;

        // Create variable to hold the line as a stream instead of a string,
        // Allows easier tokenization
        std::stringstream stream (line);
        std::string token;
        size_t featureIndex = 0;

        // Keep parsing the line stream and placing into Data
        while (std::getline (stream, token, ','))
            Data.at (lineIndex).at (featureIndex++) = token;
        lineIndex++;
    }

    inputData.close ();
    return 0;
}


int countClassTypes (const std::vector<std::vector<std::string>> &Data, std::unordered_map<std::string, size_t> &classCounts)
{
    // Count the number of occurances for each label and store the results in a map
    for (const std::vector<std::string> &row : Data)
        classCounts[row.back ()]++;
    return 0;
}


float calcEntropy (const std::unordered_map<std::string, size_t> &classCounts, const size_t rows)
{
    // First check if there are zero rows
    if (rows == 0)
        return 0.0;

    // Calculate the entropy, used by Gain and Gain Ratio
    float sum = 0;
    for (const auto &pair : classCounts)
    {
        // Safe guard against zero count entries
        if (pair.second == 0)
            continue;
        float pi = ((float)pair.second) / ((float)rows);
        sum -= pi * std::log2 (pi);
    }

    return sum;
}


float evalContGain (const std::vector<std::vector<std::string>> &Data, const size_t col, const float parEntropy, float &threshold)
{
    // If there is only one transact, the data is already full classified perfectly.
    if (Data.size () < 2)
        return 0;

    // Create a new vector of pairs of the specific column and the label that comes from it, place Data rows into it and sort the pairs
    std::vector<std::pair<float, std::string>> samples;
    samples.reserve (Data.size ());
    for (auto &row : Data)
        samples.push_back ({std::stof (row[col]), row.back ()});
    std::sort (samples.begin (), samples.end ());

    // Split the Data into two parts bounded by index i and calculate the entropy of the split
    // If gain is higher than maxGain, then set the current setup as the best setup and store the results
    std::unordered_map<std::string, size_t> leftCounts, rightCounts;
    countClassTypes (Data, rightCounts);
    float maxGain = -1.0;
    threshold = 0.0;
    for (int i = 0; i < samples.size () - 1; i++) 
    {
        rightCounts[samples[i].second]--;
        leftCounts[samples[i].second]++;
        if (samples[i].first == samples[i+1].first)
            continue;
        float mid = (samples[i].first + samples[i+1].first) / 2.0;
        float entropy = ((float)(i + 1) / samples.size ()) * calcEntropy (leftCounts, i + 1) +
                    ((float)(samples.size () - (i + 1)) / samples.size ()) * calcEntropy (rightCounts, samples.size () - (i + 1));
        float gain = parEntropy - entropy;
        if (gain > maxGain)
        {
            maxGain = gain;
            threshold = mid;
        }
    }
    return (maxGain > 0) ? maxGain : 0;
}


float evalDiscGain (const std::vector<std::vector<std::string>> &Data, const size_t col, const float parEntropy)
{
    // If there is only one transaction, there is nothing to split
    if (Data.size () < 2)
        return 0.0;

    // Create a map that stores the different types of labels for feature col and for each,
    // Create a map of the occurances of the target for them, let's say that col has Rain and Sun,
    // and the targets was play outside, it could be that the outer map has rain and sun as keys and 
    // the corresponds maps as values while the nested maps include the number of yeses and nos on whether
    // it is raining or not.
    std::unordered_map<std::string, std::unordered_map<std::string, size_t>> colMaps;
    std::unordered_map<std::string, size_t> colSize;
    
    for (const auto &row : Data)
    {
        colMaps[row[col]][row.back ()]++;
        colSize[row[col]]++;
    }

    // Check if all the rows in Data have the same feature name
    if (colMaps.size () < 2)
        return 0.0;

    // We now go through each type of label for col and determine the entropy and calculated the weighted average
    float entropy = 0;
    for (const auto &[key, value] : colMaps)
    {
        entropy += ((float)colSize[key] / Data.size ()) * calcEntropy (value, colSize[key]);
    }
    return parEntropy - entropy;
}


std::string findMajorityLabel (const std::unordered_map<std::string, size_t> &classCounts)
{
    std::string majorityLabel;
    size_t maxCount = 0;
    
    // If it is, then find the label that occurs the most frequently and return that as a leaf node
    for (const auto &pair : classCounts)
    {
        if (pair.second > maxCount)
        {
            maxCount = pair.second;
            majorityLabel = pair.first;
        }
    }
    return majorityLabel;
}


float splitInfo (const std::vector<size_t> &branchSizes, const size_t rows)
{
    if (rows == 0)
        return 0.0;

    // Calculate the splitInfo for the data set
    float sum = 0;
    for (const auto &branch : branchSizes)
    {
        if (branch == 0)
            continue;
        sum -= ((float)branch / rows) * std::log2 ((float)(branch) / rows);
    }
    return sum;
}


float evalContGainRatio (const std::vector<std::vector<std::string>> &Data, const size_t col, const float parEntropy, float &threshold)
{
    // If there is only one transact, the data is already full classified perfectly.
    if (Data.size () < 2)
        return 0;

    // Create a new vector of pairs of the specific column and the label that comes from it, place Data rows into it and sort the pairs
    std::vector<std::pair<float, std::string>> samples;
    samples.reserve (Data.size ());
    for (auto &row : Data)
        samples.push_back ({std::stof (row[col]), row.back ()});
    std::sort (samples.begin (), samples.end ());

    // Split the Data into two parts bounded by index i and calculate the entropy of the split
    // If gain is higher than maxGain, then set the current setup as the best setup and store the results
    float maxGainRatio = -1.0;
    threshold = 0.0;
    std::unordered_map<std::string, size_t> leftCounts, rightCounts;
    countClassTypes (Data, rightCounts);
    for (int i = 0; i < samples.size () - 1; i++) 
    {
        rightCounts[samples[i].second]--;
        leftCounts[samples[i].second]++;
        if (samples[i].first == samples[i+1].first)
            continue;
        std::vector<size_t> branchSizes  = {(size_t)(i + 1), samples.size () - (i + 1)};
        float splitInfoVal = splitInfo (branchSizes, samples.size ());
        float mid = (samples[i].first + samples[i+1].first) / 2.0;
        float entropy = ((float)(i + 1) / samples.size ()) * calcEntropy (leftCounts, i + 1) +
                    ((float)(samples.size () - (i + 1)) / samples.size ()) * calcEntropy (rightCounts, samples.size () - (i + 1));
        float gainRatio = (splitInfoVal > 0) ? (parEntropy - entropy) / splitInfoVal : 0;
        if (gainRatio > maxGainRatio)
        {
            maxGainRatio = gainRatio;
            threshold = mid;
        }
    }
    return (maxGainRatio > 0) ? maxGainRatio : 0;
}


float evalDiscGainRatio (const std::vector<std::vector<std::string>> &Data, const size_t col, const float parEntropy)
{
    // We can reuse the regular GAIN function to calculate the gain portion of GAIN_RATIO. We still need to calculate
    // splitInfo separately but we can divide and return the result
    float gain = evalDiscGain (Data, col, parEntropy);

    // We now need to calculate the counts of each individual feature
    std::vector<size_t> branchSizes (calcBranchSizes (Data, col));

    // Calculate splitInfo and GAIN_RATIO
    float splitInfoVal = splitInfo (branchSizes, Data.size ());
    return (splitInfoVal > 0) ? gain / splitInfoVal : 0;
}


std::vector<size_t> calcBranchSizes (const std::vector<std::vector<std::string>> &Data, const size_t col)
{
    // We now need to calculate the counts of each individual feature
    std::unordered_map<std::string, size_t> colSize;
    std::vector<size_t> branchSizes;
    for (const auto &row : Data)
        colSize[row[col]]++;
    branchSizes.reserve (colSize.size ());
    for (const auto &[key, value] : colSize)
        branchSizes.push_back (value);
    return branchSizes;
}


float calcGini (const std::unordered_map<std::string, size_t> &classCounts, const size_t rows)
{
    // First check if there are zero rows
    if (rows == 0)
        return 0.0;

    // Calculate Gini
    float sum = 0;
    for (const auto &pair : classCounts)
    {
        // Safe guard against zero count entries
        if (pair.second == 0)
            continue;
        float pi = ((float)pair.second) / ((float)rows);
        sum += pi * pi;
    }
    return 1- sum;
}


float evalContGini (const std::vector<std::vector<std::string>> &Data, const size_t col, const float parGini, float &threshold)
{
    // If there is only one transact, the data is already full classified perfectly.
    if (Data.size () < 2)
        return 0;

    // Create a new vector of pairs of the specific column and the label that comes from it, place Data rows into it and sort the pairs
    std::vector<std::pair<float, std::string>> samples;
    samples.reserve (Data.size ());
    for (auto &row : Data)
        samples.push_back ({std::stof (row[col]), row.back ()});
    std::sort (samples.begin (), samples.end ());

    // Split the Data into two parts bounded by index i and calculate the entropy of the split
    // If gain is higher than maxGain, then set the current setup as the best setup and store the results
    float maxGain = -1.0;
    threshold = 0.0;
    std::unordered_map<std::string, size_t> leftCounts, rightCounts;
    countClassTypes (Data, rightCounts);
    for (int i = 0; i < samples.size () - 1; i++) 
    {
        rightCounts[samples[i].second]--;
        leftCounts[samples[i].second]++;
        if (samples[i].first == samples[i+1].first)
            continue;
        float mid = (samples[i].first + samples[i+1].first) / 2.0;
        float gini = ((float)(i + 1) / samples.size ()) * calcGini (leftCounts, i + 1) +
                    ((float)(samples.size () - (i + 1)) / samples.size ()) * calcGini (rightCounts, samples.size () - (i + 1));
        float gain = parGini - gini;
        if (gain > maxGain)
        {
            maxGain = gain;
            threshold = mid;
        }
    }
    return (maxGain > 0) ? maxGain : 0;
}


float evalDiscGini (const std::vector<std::vector<std::string>> &Data, const size_t col, const float parGini, std::unordered_set<std::string> &leftSet)
{
    leftSet = {};
    if (Data.size () < 2)
        return 0.0;

    // Create a map that stores the different types of labels for feature col and for each,
    // Create a map of the occurances of the target for them, let's say that col has Rain and Sun,
    // and the targets was play outside, it could be that the outer map has rain and sun as keys and 
    // the corresponds maps as values while the nested maps include the number of yeses and nos on whether
    // it is raining or not.
    std::unordered_map<std::string, std::unordered_map<std::string, size_t>> colMaps;
    std::unordered_map<std::string, size_t> colSize;
    
    for (const auto &row : Data)
    {
        colMaps[row[col]][row.back ()]++;
        colSize[row[col]]++;
    }

    // Check if all the rows in Data have the same feature name
    if (colMaps.size () < 2)
        return 0.0;

    // Find the majority class label within Data
    std::unordered_map<std::string, size_t> classCounts;
    countClassTypes (Data, classCounts);
    std::string majorityLabel = findMajorityLabel (classCounts);

    // Calculate the proportion of each discrete feature of col in Data with how often
    // it has the entire Data's majority label as its target label. We store the proportion
    // and the discrete feature into sortedCategories.
    // We then sort the vector so that discrete values that have less of the majority label
    // are at the front of the vector and vice-versa.
    std::vector<std::pair<float, std::string>> sortedCategories;
    for (const auto &[key, value] : colSize)
    {
        float prop = (float)colMaps[key][majorityLabel] / value;
        sortedCategories.push_back ({prop, key});
    }
    std::sort (sortedCategories.begin (), sortedCategories.end ());
    
    // We use a sliding window principle to determine which discrete categories belond in the left set
    // (and thus, the right set by negation). We place all target classes and their counts into rightCounts
    // We then loop through sortedCategories and remove
    // that category from rightCounts and place into leftCounts. We then determine the weighted average
    // gini of that split and see if it is the best cut we have seen before. If it is, we store it so
    // we can place it into leftSet at the end.
    std::unordered_map<std::string, size_t> leftCounts, rightCounts (classCounts);
    float maxGain = -1.0;
    size_t leftRows = 0, rightRows = Data.size ();
    int bestCut = -1;
    for (size_t i = 0; i < sortedCategories.size () - 1; i++)
    {
        // find the new category that we are going to move over and note the number of row changes
        std::string category = sortedCategories[i].second;
        leftRows += colSize[category];
        rightRows -= colSize[category];

        // for leftCounts and rightCounts, change the count of target label is has for the specific category
        for (const auto &[key, value] : colMaps[category])
        {
            leftCounts[key] += value;
            rightCounts[key] -= value;
        } 
        
        // Calculate gini
        float gain = parGini - (((float)leftRows / (leftRows + rightRows)) * calcGini (leftCounts, leftRows)
                                +((float)rightRows / (leftRows + rightRows)) * calcGini (rightCounts, rightRows));
        if (gain > maxGain)
        {
            maxGain = gain;
            bestCut = i;
        }
    }

    // Now, fill leftSet before returning
    if (maxGain > 0)
        for (int i = 0; i <= bestCut; i++)    
            leftSet.insert (sortedCategories[i].second);
    return (maxGain > 0) ? maxGain : 0; 
}


float calcBestGain (const bool cont, const float parentG, const splitting criterion, const size_t col, float &threshold, 
    std::unordered_set<std::string> &leftThreshold, const std::vector<std::vector<std::string>> &Data)
{
    switch (criterion)
    {
        case GAIN:
            return cont ? evalContGain (Data, col, parentG, threshold)
                        : evalDiscGain (Data, col, parentG);
            break;
        case GAIN_RATIO:
            return cont ? evalContGainRatio (Data, col, parentG, threshold)
                        : evalDiscGainRatio (Data, col, parentG);
            break;
        case GINI:
            return cont ? evalContGini (Data, col, parentG, threshold)
                        : evalDiscGini (Data, col, parentG, leftThreshold);
            break;
        default:
            std::cout << "Invalid criterion." << std::endl;
            exit (1);
    }
    return 0;
}