#ifndef TREE_NODE
#define TREE_NODE

#include <unordered_map>
#include <string>
#include <vector>
#include <unordered_set>

#include "funcDef.hpp"

class node
{
    public:
        node (const std::string names, const std::string label, const bool cont);
        ~node ();
        std::unordered_map<std::string, node*> children;
        std::unordered_set<std::string> leftBranchFeatures, rightBranchFeatures;
        std::string featureName;
        std::string classLabel;
        std::string majorityClass;
        bool continuous;
        float contBound;
        bool isLeaf () const { return children.empty (); }
};


class tree
{
    public:
        tree (const std::vector<std::string> &names, const std::vector<bool> &cont, const splitting &criteria);
        ~tree ();
        int train (const std::vector<std::vector<std::string>> &Data);
        std::string predict (const std::vector<std::string> &object);
    private:
        node* root;
        splitting criterion;
        std::vector<bool> contVector;
        std::vector<std::string> featureNames;
        node* buildTree (const std::vector<std::vector<std::string>> &Data, std::vector<int> attributes);
};


#endif
