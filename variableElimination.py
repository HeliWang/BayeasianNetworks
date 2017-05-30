import numpy as np

def restrict(factor, variable, value):
    """function that restricts a variable to some value in a given factor"""
    slc = [slice(None)] * len(factor.shape)
    slc[variable] = value
    res = factor[slc]
    
    #reshape
    shapeLst = list(factor.shape)
    shapeLst[variable] = 1
    res = res.reshape(tuple(shapeLst))
    
    return normalize(res)

def multiply(factor1, factor2):
    """function that multiplies two factors"""
    return normalize(factor1 * factor2);

def sumout(factor, variable):
    """function that sums out a variable in a given factor"""
    res = factor.sum(axis=variable);
    shapeLst = list(factor.shape)
    shapeLst[variable] = 1
    res = res.reshape(tuple(shapeLst))
    return normalize(res);
    
def normalize(factor):
    """function that normalizes a factor by dividing each entry by the sum of all the entrie"""
    factor = factor * (1/factor.sum())
    return factor
    
def inference(factorList, queryVariables, orderedListOfHiddenVariables, evidenceList):
    """function that computes Pr(queryV ariables|evidenceList) by variable elimination"""
    """ This function should restrict the factors in factorList according to the evidence in evidenceList.
        Next,  it should sumout the hidden variables from the product of the factors in factorList. 
        The variables should be summed out in the order given in orderedListOfHiddenVariables. 
        Finally, the answer should be normalized when a probability distribution that sums up to 1 is desired. """
    for evidence in evidenceList:
        for idx, factor in enumerate(factorList):
            shapeLst = list(factor.shape)
            if shapeLst[evidence['var']] == 2:
                res = restrict(factor, evidence['var'], evidence['value'])
                factorList[idx] = res
        #print 'Restriction Complete: ' + str(evidence['var'])
        #print factorList
    
    for hidVarIdx, hidVar in enumerate(orderedListOfHiddenVariables):
        factorLstToMultiply = []
        factorLstIndexLst = []
        
        for idx, factor in enumerate(factorList):
            shapeLst = list(factor.shape)
            if shapeLst[hidVar] == 2:
                factorLstToMultiply.append(factor)
                factorLstIndexLst.append(idx)
        
        factorLstIndexLst.sort(reverse=True)
        
        for factorLstIndex in factorLstIndexLst:
            factorList.pop(factorLstIndex)
        
        #print 'factor List'
        #print factorList
        
        if len(factorLstToMultiply) != 0:
            sum = factorLstToMultiply.pop()
            for idx, factor in enumerate(factorLstToMultiply):
                sum = multiply(sum, factor)
            sum = sumout (sum, hidVar)
            #print 'multiply + sum-out var ' + str(hidVar)
            #print sum
            factorList.append(sum)
    
    #print 'debug'
    #print factorList
    
    # The remaining factors refer only to the query variables Q, take their product and normalize to produce P(Q)
    sum = factorList.pop()
    for idx, factor in enumerate(factorList):
        sum = multiply(sum, factor)
        
    return sum
    
def test():
    unitTest = False
    inferenceTest = True
    
    if unitTest == True:
        print("Test Restriction, Normalization")
        
        #h(a, b, c)
        factor1 = np.array([[[0.12, 0.48], [0.12, 0.28]], [[0.02, 0.08],[0.27, 0.63]]])
        
        #restrict b -> 1 
        #aka new evidence +b
        variable1 = 1;
        value1 = 1;
        
        result1 = restrict(factor1, variable1, value1)
        print(result1)
        
        print("Test Multiplication")
        
        #h(a, b, c)
        factor2 = np.array([[0.6, 0.4], [0.1, 0.9]])
        factor3 = np.array([[0.2, 0.8], [0.3, 0.7]])
        factor2 = factor2.reshape(2,2,1)
        factor3 = factor3.reshape(1,2,2)
        #print (factor1)
        print (multiply (factor2, factor3))
    
        print("Test Summing Out of a Factor1")
        factor4 = np.array([[0.6, 0.4], [0.1, 0.9]])
        factor4 = factor4.reshape(2,2,1)
        #Summing out variable a (0)
        variable0 = 0;
        print (sumout (factor4, variable0))
     
        print("Test Summing Out of a Factor2")
        factor5 = np.array([[[[[[ 0.172532  ]]], [[[ 0.006929  ]]]]], [[[[[ 0.81233361]]], [[[ 0.00820539]]]]]])
        #Summing out variable a (0)
        variable0 = 0;
        print (sumout (factor5, variable0))
     
    
    if inferenceTest == True:
        print("Test Inference Logic - Pr(fraud|fp, ~ip, crp)")
        factorList = []
        queryVariables = []
        orderedListOfHiddenVariables = []
        evidenceList = []
        # Factor Format: Trav,OC,Fraud,CRP,IP,FP
        
        # P1 = Pr(CRP|OC), P2 = Pr(CRP|~OC)
        P1 = 0.1
        P2 = 0.001
        factorList.append(np.array([[1-P2, P2], [1-P1, P1]]).reshape(1,2,1,2,1,1))
    
        # P3 = Pr(Trav)
        P3 = 0.05
        factorList.append(np.array([1-P3, P3]).reshape(2,1,1,1,1,1))
    
        # P4 = Pr(FP|~Trav,Fraud) P5 = Pr(FP|~Trav,~Fraud) P6 = Pr(FP|Trav,Fraud) P7 = Pr(FP|Trav,~Fraud)
        P4 = 0.1
        P5 = 0.01
        P6 = 0.9
        P7 = 0.9
        factorList.append(np.array([[[1-P5,P5], [1-P4,P4]],[[1-P7,P7],[1-P6,P6]]]).reshape(2,1,2,1,1,2))
        
        # P8 = Pr(IP|OC,~Fraud) P9 = Pr(IP|OC,Fraud) P10 = Pr(IP|~OC,~Fraud) P11 = Pr(IP|~OC,Fraud)
        P8  = 0.01
        P9  = 0.02
        P10 = 0.001
        P11 = 0.011
        factorList.append(np.array([[[1-P10,P10], [1-P11,P11]],[[1-P8,P8],[1-P9,P9]]]).reshape(1,2,2,1,2,1))
    
        # P12 = Pr(OC)
        P12 = 0.7
        factorList.append(np.array([1-P12, P12]).reshape(1,2,1,1,1,1))
    
        # P13 = Pr(Fraud|Trav) P14 = Pr(Fraud|~Trav)
        P13 = 0.01
        P14 = 0.004
        factorList.append(np.array([[1-P14, P14], [1-P13, P13]]).reshape(2,1,2,1,1,1))
        
        # queryVariables = [Fraud]
        queryVariables.append(2)
        
        # evidenceList = ['+fp', '~IP', '+CRP']
        evidenceList.append({'var':5, 'value':1})
        evidenceList.append({'var':4, 'value':0})
        evidenceList.append({'var':3, 'value':1})
        
        # queryVariables = ['Trav','OC']
        orderedListOfHiddenVariables.append(0)
        orderedListOfHiddenVariables.append(1)
    
        print inference(factorList, queryVariables, orderedListOfHiddenVariables, evidenceList)
   
        
test()