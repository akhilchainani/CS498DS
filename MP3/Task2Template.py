
# coding: utf-8

import opengm
import numpy as np

import sys
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s] (%(threadName)-10s) %(message)s',
                    stream=sys.stdout)
logging.info("Imports successful")

class FactorGraph(object):
    
    def __init__(self):
        logging.debug("Created Factor Graph Object")
    
    def infer(self):
        logging.info("Running Inference")
        ###########################
        # define binary variables #
        ###########################
        # two binary variables: 
        #var 0 (stage S1): dimension is 2
        #var 1 (event E1): dimension is 2
        variables = [2,2]
	
        ################################
        # # construct the Factor Graph #
        ################################
        gm = opengm.graphicalModel(variables, operator='multiplier')

        ########################################################################################
        # TODO: Fill in values in g_func, and g_var, according to the provided tables #
        ########################################################################################
        f_func = np.array([0.1, 0.9])   # priors f
        f_var = [0]                     # f(S1) using S1 as variable 0
        g_func = np.array([[0,0.2],
			   [0,0.5]])    # factor function g
        g_var = [0,1]                     # g(S1, S2)
        ############
        # END TODO #
        ############


        ##################################
        # connect factor functions to FG #
        ##################################

        gm.addFactor(gm.addFunction(f_func),f_var) # add prior to event
        gm.addFactor(gm.addFunction(g_func),g_var) # add factor function to event (E1) and stage (S1)


        ##################################
        # # belief propagation inference #
        ##################################
        inf=opengm.inference.BeliefPropagation(gm,accumulator='maximizer')
        inf.infer()

        ##############
        # get argmax #
        ##############

        arg=inf.arg()
        print("Inference result: ", arg)

        ##############################
        # get marginal probabilities #
        ##############################
        marginals = inf.marginals(range(len(variables)))

        #############################################################
        # # get marginal of the state variable (variable index = 0) #
        #############################################################
        vars = [0]
	max_marg = marginals[0]
	max_val = 0
        for i in vars:
            marginals_xi = marginals[i]
	    if marginals_xi > max_marg:
		max_val = i
		max_marg = marginals_xi
            marginals_xi /= np.sum(marginals_xi)
            print("x_{} marginal: {}".format(i, marginals_xi))
        pass

m = FactorGraph()
m.infer()

