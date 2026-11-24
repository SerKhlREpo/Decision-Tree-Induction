#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>


#include "treeNode.hpp"
#include "funcDef.hpp"


node::node (const std::string names, const std::string label, const bool cont)
{
    // Assign features that are known to the node
    // contBound is assigned later as it isn't known at the time of creation.
    featureName = names;
    classLabel = label;
    continuous = cont;
    contBound = 0.0;
}


node::~node ()
{
    // Delete all of the children of this node
    for (auto &child : children)
        delete child.second;
}


tree::tree (const std::vector<std::string> &names, const std::vector<bool> &cont, const splitting &criteria)
{
    // Store the feature names, which are discrete/continuous, and the splitting criterion into the class
    criterion = criteria;
    contVector = cont;
    featureNames = names;

    // Declare the root as a nullptr for now
    root = nullptr;
}


tree::~tree ()
{
    // Delete the root and the node will handle the recursive cleaning of the tree.
    delete root;
}


node* tree::buildTree (const std::vector<std::vector<std::string>> &Data, std::vector<int> attributes)
{
    // Check for the case where the (sub)set is empty
    if (Data.empty ())
        return nullptr;

    // Check for the case if the remaining subset has just one class type
    std::unordered_map<std::string, size_t> classCounts;
    countClassTypes (Data, classCounts);
    if (classCounts.size () == 1)
    {
        // If all other results yield the same class definintion, create a leaf node and return it
        node* leaf = new node ("", classCounts.begin ()->first, true);
        return leaf;
    }

    // Check for if the attribute list is empty
    if (attributes.empty ())
    {
        node* leaf = new node ("", findMajorityLabel (classCounts), true);
        return leaf;
    }

    // Recursive splitting of the Data
    float parentEntropy = calcEntropy (classCounts, Data.size ());
    float parentGini = calcGini (classCounts, Data.size ());
    float bestThreshold = 0;
    float bestGain = -1.0;
    int bestCol = -1;
    std::unordered_set<std::string> bestLeftThreshold, bestRightThreshold;

    // Go through all of the splitting features in Data
    for (int col : attributes)
    {
        // Calculate the best possible gain for splitting on that feature
        std::unordered_set<std::string> leftThreshold;
        float threshold = 0.0;
        float gain = calcBestGain (contVector[col], (criterion == GINI) ? parentGini : parentEntropy, 
                                    criterion, col, threshold, leftThreshold, Data);
        if (gain > bestGain)
        {
            bestGain = gain;
            bestCol = col;
            if (criterion == GINI)
                bestLeftThreshold = leftThreshold;
            bestThreshold = threshold;
        }
    }

    // if no best column was found, just return a leaf node that is the majority target class
    if (bestCol < 0 || bestGain <= 0)
        return new node ("", findMajorityLabel (classCounts), true);

    // Now that we have the split, split the Data into their corresponding subset and recursively call this function.
    if (contVector[bestCol])
    {
        // separate Data into rows with one being less than threshold and other being higher
        std::vector<std::vector<std::string>> leftData, rightData;
        for (const auto &row : Data)
        {
            if (std::stof (row[bestCol]) <= bestThreshold)
                leftData.push_back (row);
            else
                rightData.push_back (row);
        }

        // Check if either side (left/right) is empty, if it is, then return the majority target class
        if (leftData.empty () || rightData.empty ())
            return new node ("", findMajorityLabel (classCounts), true);

        // Create a new node (this level is parent)
        node* interNode = new node (featureNames[bestCol], "", true);
        interNode->contBound = bestThreshold;
        interNode->majorityClass = findMajorityLabel (classCounts);

        // recursively call this function to create the children and append to the children map of this level node
        interNode->children.insert ({"<=" + featureNames[bestCol], buildTree (leftData, attributes)});
        interNode->children.insert ({">" + featureNames[bestCol], buildTree (rightData, attributes)});
        return interNode;
    } else
    {
        // Remove the attribute we just split on just avoiding splitting on it again
        if (criterion != GINI)
            attributes.erase (std::find (attributes.begin (), attributes.end (), bestCol));

        // Create all of the different subsets of Data for the different labels for bestCol
        std::unordered_map<std::string, std::vector<std::vector<std::string>>> dataSubsets;
        for (const auto &row : Data)
        {
            if (criterion == GINI)
            {
                if (bestLeftThreshold.count (row[bestCol]) > 0)
                    dataSubsets["left"].push_back (row);
                else
                {
                    dataSubsets["right"].push_back (row);
                    bestRightThreshold.insert (row[bestCol]);
                }
            } else
                dataSubsets[row[bestCol]].push_back (row);
        }

        // Now that we have the Data subsets, we need to recursively call this function on them and 
        // append the nodes they return as children to the parent node (this level)
        node* interNode = new node (featureNames[bestCol], "", false);
        interNode->majorityClass = findMajorityLabel (classCounts);
        if (criterion == GINI)
        {
            interNode->leftBranchFeatures = bestLeftThreshold;
            interNode->rightBranchFeatures = bestRightThreshold;
        }
        for (const auto &[key, value] : dataSubsets)
            interNode->children.insert ({key, buildTree (value, attributes)});
        return interNode;
    }

    return nullptr;
}


int tree::train (const std::vector<std::vector<std::string>> &Data)
{
    // Check if the Data vector is empty (failed to read input file)
    if (Data.empty ())
        exit (1);

    // Populate attributes with the features. 
    std::vector<int> row (Data[0].size () - 1);
    std::iota (row.begin (), row.end (), 0);

    root = buildTree (Data, row);
    return 0; 
}


std::string tree::predict (const std::vector<std::string> &object)
{
    // We first check if the tree is empty
    if (root == nullptr)
        return "NULL";
    
    // We then will trace down the tree until we reach a leaf node, returning NULL if we get a feature
    // that wasn't seen before
    node* curr = root;
    while (curr != nullptr && !curr->isLeaf ())
    {
        // Pull information on the specific feature we are splitting and which column it is on
        std::string name = curr->featureName;
        auto it = std::find (featureNames.begin (), featureNames.end (), name);
        size_t col = std::distance (featureNames.begin (), it);
        if (it == featureNames.end () || col >= object.size ())
            return "NULL";

        // Split cases on whether it is continuous or not
        if (contVector[col] == true)
        {
            if (std::stof (object[col]) <= curr->contBound)
                curr = curr->children["<=" + name];
            else
                curr = curr->children[">" + name];
        } else
        {
            // Split the case between GINI and the rest
            if (criterion == GINI)
            {
                if (curr->leftBranchFeatures.find (object[col]) != curr->leftBranchFeatures.end ())
                    curr = curr->children.at ("left");
                else if (curr->rightBranchFeatures.find (object[col]) != curr->rightBranchFeatures.end ())
                    curr = curr->children.at ("right");
                else
                    return curr->majorityClass;
            } else
            {
                if (curr->children.find (object[col]) != curr->children.end ())
                    curr = curr->children.at (object[col]);
                else
                    return curr->majorityClass;
            }
        }
    }
    if (curr == nullptr)
        return "NULL";
    // We reached a leaf node
    return curr->classLabel;
    
}