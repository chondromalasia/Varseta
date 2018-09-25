# -*- coding: utf-8 -*-

class Evaluation:
    
    def __init__(self, variation_sets, gold_sets):
        self.variation_sets = variation_sets
        self.gold_sets = gold_sets
        self.strict_precision, self.strict_recall, self.strict_f1 = self.strict_evaluate()
        self.fuzzy_precision, self.fuzzy_recall, self.fuzzy_f1 = self.fuzzy_evaluate()

    def strict_evaluate(self):
        tp=0
        for elem in self.gold_sets:
            try:
                if elem in self.variation_sets:
                    tp+=1
            except:
                pass
        fp=(len(self.variation_sets))-tp
        fn=(len(self.gold_sets))-tp
        p=float(tp/(tp+fp+0.0001))
        r=float(tp/(tp+fn+0.0001))
        f=float(2*(p*r)/(p+r+0.0001))

        return p, r, f
    
    
    def fuzzy_evaluate(self):
        tp=0
        l_var=len(self.variation_sets)
        l_gold = len(self.gold_sets)
        found = False
        for elem in self.variation_sets:
            try:
            
# can be added if for some reason distinction between exact and fuzzy overlap is necessary in fuzzy matchin            
#                if elem in self.gold_sets:
#                    tp+=1
#                   self.gold_sets.remove(elem)
                    
#                else:
                for x in range(len(elem)):
                	for s in self.gold_sets:
                		if elem[x] in s:
                			tp+=1
         	     			# needs one overlap per set for a true positive
            				self.gold_sets.remove(s)
                			found = True
                			break
                			
            		# break out of the double loop	
               		if found == True:
               			found = False
               			break 
            except:
                pass

        fp=l_var-tp
        fn=l_gold-tp
        p=float(tp/(tp+fp+0.0001))
        r=float(tp/(tp+fn+0.0001))
        f=float(2*(p*r)/(p+r+0.0001))
        
        return p, r, f

