'''
Created on Apr 8, 2016

@author: jesus calvillo
Contains methods used to load and pre-process the corpus 
'''

import numpy as np
import theano

from tools.similarities import binaryEquality
from containers import Situation,TrainingElement,InputObjectAP,Corpus,CorpusAP
from crossValidation import Fold


rawPrologFilePath='dataFiles/rawPrologOutput/model.prod_train.localist.set'
wordLocalistMapPath='dataFiles/map_localist_words.txt'
dsssMatrixPath="dataFiles/model_vectors"
       
       
'''
Loads the file in wordLocalistMapPath, creating a dictionary of words to localist vectors.
'''       
def getWordLocalistMap(filename):
    FILE=open(filename,'r')
    
    indexToWord={}
    for line in FILE:
        segs=line.split('[')
        word=segs[0][:-1]
        
        vector=segs[1].strip()
        vector=vector[:-1]
        vector=vector.split(",")
        vector=[float(i) for i in vector]
        index=np.argmax(vector,axis=0)  
        indexToWord[index]=word    
    FILE.close()  

    return indexToWord

'''
Loads the file in dsssMatrixPath and returns a matrix concatenating all situation vectors of the atomic events
Returns also a list containing all basic events in the file
'''
def getAtomicEventDSSMap(filename):
    FILE=open(filename,'r')
    
    dsssMatrix=[]
    events=[] 
    for line in FILE:
        segments=line.split()
        predicate=segments[0]
        dss=segments[1:]
        dss=np.asarray(dss).astype(theano.config.floatX)  # @UndefinedVariable
        dsssMatrix.append(dss)
        events.append(predicate)
    dsssMatrix=np.asarray(dsssMatrix)
    return dsssMatrix,events

'''
Returns a map Word->Index, by taking a map Index->Word
'''
def getMapWordToIndex(mapIndexWord):
    mapWordToIndex={}
    for key,value in mapIndexWord.iteritems():
        mapWordToIndex[value]=key    
    return mapWordToIndex

'''
Takes a list of TrainingElement instances and appends the localist representation of a period to wordsLocalist
'''
def addPeriods(trainingElementList,vocabSize):
    dot=[0]*vocabSize
    dot.append(1)
    for item in trainingElementList:
        if not hasattr(item, 'period')or item.period==False:
            for i in xrange(len(item.wordsLocalist)):
                item.wordsLocalist[i]=np.append(item.wordsLocalist[i],[0])
            item.wordsLocalist.append(dot)
            item.period=True
            item.testItem=item.testItem+ " ."
'''
Takes a list of TrainingElement instances and sets its input type
'''
def setInputType(trainingElementList,inputType):
    if inputType=="dss":
        for sent in trainingElementList: sent.input=sent.dss150
    if inputType=="sitVector":
        for sent in trainingElementList: sent.input=sent.sitVector
    if inputType=="compVector":
        for sent in trainingElementList: sent.input=sent.compVector
    if inputType=="beliefVector":
        for sent in trainingElementList: sent.input=sent.DSSValue
'''
Takes a list of TrainingElement instances and sets its input type
'''
def setOutputType(trainingElementList,outputType):
    if outputType=="dss":
        for sent in trainingElementList: sent.output=sent.dss150
    if outputType=="sitVector":
        for sent in trainingElementList: sent.output=sent.sitVector
    if outputType=="compVector":
        for sent in trainingElementList: sent.output=sent.compVector
    if outputType=="beliefVector":
        for sent in trainingElementList: sent.output=sent.DSSValue



#############################################################################################################
#### CONDITIONAL PROBABILITY SCORES AND COMPREHENSION SCORES
#############################################################################################################

def getConditionalProbs(dssVector,dsssMatrix):
    dotp=np.dot(dsssMatrix,dssVector)      
    return dotp/np.sum(dssVector)   
'''
Takes a list of TrainingElement instances (that already have beliefVector), the DSS situation space matrix, and its dimensionality (150 or 25000)
Returns the same list but adding the comprehension scores according to Frank et al. (2009)'s paper
'''
def getComprehensionVectorsBelief(trainingElementList,eventMatrix,dimensionality):
    eventPriors=np.sum(eventMatrix,axis=1)/dimensionality
    
    for item in trainingElementList:
        item.compVector=item.beliefVector-eventPriors
        compVector=[];
        for elem,posterior,prior in zip(item.compVector,item.beliefVector,eventPriors):
            compScore=0.0;
            if posterior>prior:
                compScore=elem/(1.0-prior)
            else:
                compScore=elem/prior
            compVector.append(compScore)
        item.compVector=np.asarray(compVector)

#############################################################################################################
#### LOAD AND OBTAIN CORPUS FROM RAW PROLOG-OUTPUT FILES
#############################################################################################################

'''
Takes a file containing the output of the prolog file with dss-sentences and the full 25K situation vectors
Returns a list of TrainingElement instances, where each of the latter is a sentence couple with all its information
It computes the belief vector directly and puts it into each TrainingElement
'''
def loadAndConvertRawCorpus_belief(inputPath, matrixPath):
    trainingCorpus=[]
    dsssMatrixBelief,_= getAtomicEventDSSMap(matrixPath)
      
    with open(inputPath,'rb') as inputFile:
        trainingItem=0
        while True:
            line=inputFile.readline()
            if not line: 
                if trainingItem!=0:
                    trainingCorpus.append(trainingItem)
                break
            if((len(line.split())>0)and(line.split()[0]=='Item')):
                headerSegs= line.split("\"")
                testItem=headerSegs[1]
                numberOfWords=int(headerSegs[2])
                semantics=headerSegs[3]
                 
                if trainingItem!=0:
                    trainingCorpus.append(trainingItem)
                 
                dss=[]
                beliefVector=[]
                wordsLocalist=[]
                trainingItem=TrainingElement(testItem,numberOfWords,semantics,wordsLocalist,dss)
                 
            if((len(line.split())>0)and(line.split()[0]=='Input')):
                segs= line.split()
                 
                if len(beliefVector)==0:
                    beliefVector= segs[1:25001]
                    beliefVector= np.asarray(beliefVector).astype(theano.config.floatX)  # @UndefinedVariable
                    dotp=np.dot(dsssMatrixBelief,beliefVector)      
                    beliefVector=dotp/np.sum(beliefVector)
                    trainingItem.beliefVector=beliefVector
 
                wordLocalist= segs[25002:]
                wordLocalist=[int(round(float(i))) for i in wordLocalist]
                wordLocalist=np.asarray(wordLocalist).astype('int8')
                 
                wordsLocalist.append(wordLocalist)
                 
    return trainingCorpus

'''
Takes a file containing the output of the prolog file with dss-sentences 
Returns a list of TrainingElement instances, where each of the latter is a sentence couple with all its information

Depending on the size of the DSS representation (vectorSize) obtained from the prolog file:
150: 150-dimensional dss vectors obtained after the dimensionality reduction
25K: full situation vectors with no dimensionality reduction
10:  situations that are impossible in the microworld, in this case the vectors contains only zeros

Testing with Impossible events
    # corpus=loadRawCorpus_VectorSize("dataFiles/filesWith0P/model.prod_train_passive.localist_p0.set",10,False)
    # for elem in corpus:
    #     print elem.testItem
'''
def loadRawCorpus_VectorSize(inputPath,vectorSize,active):
    trainingCorpus=[]
    
    with open(inputPath,'rb') as inputFile:
        trainingItem=0
        while True:
            line=inputFile.readline()
            if not line: 
                if trainingItem!=0:
                    trainingCorpus.append(trainingItem)
                break
            
            if((len(line.split())>0)and(line.split()[0]=='Item')):
                headerSegs= line.split("\"")
                schema=headerSegs[0].split()[1]
                testItem=headerSegs[1]
                numberOfWords=int(headerSegs[2])
                semantics=headerSegs[3]
                
                if trainingItem!=0:
                    trainingCorpus.append(trainingItem)
                
                semanticVector=[]
                wordsLocalist=[]
                trainingItem=TrainingElement(schema,testItem,numberOfWords,semantics,wordsLocalist,semanticVector,active)
                
            if((len(line.split())>0)and(line.split()[0]=='Input')):
                segs= line.split()
                
                if len(semanticVector)==0:
                    semanticVector= segs[1:vectorSize+1]
                    semanticVector= np.asarray(semanticVector).astype(theano.config.floatX)  # @UndefinedVariable
                    trainingItem.DSSValue=semanticVector

                wordLocalist= segs[vectorSize+2:]
                wordLocalist=[int(round(float(i))) for i in wordLocalist]
                wordLocalist=np.asarray(wordLocalist).astype('int8')
                
                wordsLocalist.append(wordLocalist)
                
    addPeriods(trainingCorpus,42) #add periods to the sentences, 42 is the vocabulary size
    return trainingCorpus


'''
Takes a list of TrainingElement instances, obtained from loadRoadCorpus_VectorSize, which contains either only
active or passive sentences
Puts all semantically equivalent sentences into one InputObjectAP, which is also put into a list
'''
def getCollapsedCorpus25K(normalCorpus,active):
    collapsedCorpus=[]
    for trainingElement in normalCorpus:
        match=False
        for item in collapsedCorpus:
            equal= binaryEquality(trainingElement.DSSValue,item.value)
            if equal:
                match=True
                item.sentences.append(trainingElement)
                break
        if not match:
            newItem=InputObjectAP(trainingElement.DSSValue,active)
            newItem.sentences.append(trainingElement)
            collapsedCorpus.append(newItem)
    return collapsedCorpus


'''
Loads the active and passive sentences from the prolog-output files forming lists of TrainingElement
To the loaded sentences, it adds the 150-dimensional dss
Puts together sentences with equivalent semantics and saves the resulting corpora to file
Each file contains either only active or only passive sentences.
'''
def getRawCorporaAP25KWith150DSS(prologActPath,prolog150ActPath,prologPasPath,prolog150PasPath,tag):
    
    def put150DSSinto25K(corpus25Kte,corpus150te):
        for elem in corpus25Kte:
            for elem150 in corpus150te[:]:
                if elem.testItem==elem150.testItem:
                    elem.dss150=elem150.DSSValue
                    corpus150te.remove(elem150)
        return corpus25Kte
    
    ###Active Sentences
    corpusAct25K=loadRawCorpus_VectorSize(prologActPath,25000,True)
    corpusAct150=loadRawCorpus_VectorSize(prolog150ActPath,150,True)
    corpusAct25K=put150DSSinto25K(corpusAct25K,corpusAct150)
    
    corpusActClustered=getCollapsedCorpus25K(corpusAct25K,True) #returns a list of InputObjectAP{value, ap, sentences[]}
    print len(corpusActClustered)
    
    corpusObjectAct=Corpus(corpusActClustered)
    #corpusObjectAct.saveToPickle("corpusActive25KClustered_"+tag+".pick")
    
    ###Passive Sentences
    corpusPas25K=loadRawCorpus_VectorSize(prologPasPath,25000,False)
    corpusPas150=loadRawCorpus_VectorSize(prolog150PasPath,150,False)
    corpusPas25K=put150DSSinto25K(corpusPas25K,corpusPas150)
    
    corpusPasClustered=getCollapsedCorpus25K(corpusPas25K,False) #returns a list of InputObjectAP{value, ap, sentences[]}
    print len(corpusPasClustered)
    
    corpusObjectPas=Corpus(corpusPasClustered)
    #corpusObjectPas.saveToPickle("corpusPassive25KClustered_"+tag+".pick")

    return corpusObjectAct.elements,corpusObjectPas.elements

'''
Takes two lists, one containing all IntputObjectAP related to active sentences and one related to passive sentences
Puts together the InputObjectAPs into a Situation object if they have equivalent DSSs 
Creates 2 lists of Situations: the first one corresponds to all the dss that contain both actives and passives
the second corresponds to dsss that only have actives
With that, creates a CorpusAP object and saves it to File
'''
def collapseActPasIOs_ToSituationsInCorpusAP(activeIOs,passiveIOs):
    APSits=[]
    ASits=[]
    
    for dssp in passiveIOs:
        for dssa in activeIOs[:]:
            if binaryEquality(dssp.value,dssa.value):
                newSit=Situation(dssp.value,dssa.sentences,dssp.sentences)
                APSits.append(newSit)
                activeIOs.remove(dssa)
    
    for dssa in activeIOs[:]:
        newSit=Situation(dssa.value,dssa.sentences)
        ASits.append(newSit)
        
    corpusAP = CorpusAP(APSits,ASits)
    
    return corpusAP


'''
Takes a CorpusAP object in which the Situation's sit.value is equal to the 25K situation vector
Takes also the original 25Kx44 DSS matrix
Computes the belief vectors and uses them to replace the sit.value's and the trainElem.DSSValue
Saves it to file
'''
def convertCorpusAP25KToBelief(corpusAP25K,dsssMatrix):
    def beliefVect(vector25K):
        dotp=np.dot(dsssMatrix,vector25K)
        return dotp/np.sum(vector25K)
        
    for sit in corpusAP25K.actpas:
        sit.value=beliefVect(sit.value)
        for trainElem in sit.actives:
            trainElem.DSSValue=sit.value
        for trainElem in sit.passives:
            trainElem.DSSValue=sit.value
    
    for sit in corpusAP25K.act:
        sit.value=beliefVect(sit.value)
        for trainElem in sit.actives:
            trainElem.DSSValue=sit.value
    
    
    return corpusAP25K

'''
Takes a CorpusAP object with 150 dss vectors and ads the active/passive bit to the 150-dimensional dss
Since the clustering is done using the 25K vectors, it's possible that sentences related to the same situation 
have different 150-dss vectors, because of that the bit-appending is done per-sentence
'''
def addAPBitCorpusAP25Kto150DSS(corpusAP25K):
    for sit in corpusAP25K.actpas:
        for item in sit.actives:
            item.dss150=np.append(item.dss150,1.0)
        for item in sit.passives:
            item.dss150=np.append(item.dss150,0.0)
    
    for sit in corpusAP25K.act:
        for item in sit.actives:
            item.dss150=np.append(item.dss150,1.0)
    
    return corpusAP25K

'''
Takes a CorpusAP (with belief vectors but not necessarily) and adds the active/passive bit
'''
def addAPBitCorpusAP25K(corpusAP25K):
    def addActPasBit(vector):
        actdss=np.append(vector,1.0)
        pasdss=np.append(vector,0.0)
        return actdss,pasdss
    
    for sit in corpusAP25K.actpas:
        actdss,pasdss=addActPasBit(sit.value)
        for item in sit.actives:
            item.DSSValue=actdss
        for item in sit.passives:
            item.DSSValue=pasdss
            
    for sit in corpusAP25K.act:
        actdss,pasdss=addActPasBit(sit.value)
        for item in sit.actives:
            item.DSSValue=actdss
    
    return corpusAP25K

'''
Takes a CorpusAP and sets up the item.equivalents variable of the TrainingElement instances
'''
def setupEquivalentsAP25K(corpusAP25K):
    for sit in corpusAP25K.actpas:
        for item in sit.actives:
            item.equivalents=sit.actives
        for item in sit.passives:
            item.equivalents=sit.passives
            
    for sit in corpusAP25K.act:
        for item in sit.actives:
            item.equivalents=sit.actives
 
'''
Calculates the prior probability of each sentence production rule (schema) in the trainingSet
'''
def getSchemaPriors(trainingSet):
    counts={}
    for x in xrange(1,52):
        counts[x]=0

    for elem in trainingSet:
        counts[int(elem.schema)]+=1    
    
    probs={}
    for x in xrange(1,52):
        probs[x]=counts[x]*1.0/len(trainingSet)
        
    return probs

'''
Takes the raw prolog-output files and creates a CorpusAP object. This object includes belief vectors and 150-dim DSS vectors
Also derivation length vectors

activeRawFile,passiveRawFile: Paths to the prolog-output files corresponding to the actives and passives respectively
dsssMatrixPath:               Path to the 25K-dimensional DSS Matrix
tag:                          Tag to be used to name the files
act150path,pas150path         Paths to the corresponding prolog-output files but with 150-dimesional DSSs
'''
def processCorpusAP25KFromRaw(activeRawFile,passiveRawFile,dsssMatrixPath,tag,act150path,pas150path):
    matrix,_=getAtomicEventDSSMap(dsssMatrixPath)
     
    activeIOs,passiveIOs=getRawCorporaAP25KWith150DSS(activeRawFile,act150path,passiveRawFile,pas150path,tag) 
    #2 FILES NOT! CREATED: "corpusActive25KClustered_"+tag+".pick","corpusPassive25KClustered_"+tag+".pick"
    print "RAW CORPORA LOADED... PASSIVES AND ACTIVES SEPARATED"
    
    corpusAP25K=collapseActPasIOs_ToSituationsInCorpusAP(activeIOs,passiveIOs)
    #corpusAP25K.saveToPickle("corpusAP25K_"+tag+".pick")
    print "ACTIVES AND PASSIVES CLUSTERED AND COLLAPSED INTO A CorpusAP"
    
    corpusAPFinal=convertCorpusAP25KToBelief(corpusAP25K,matrix)
    print "CorpusAP CONVERTED TO BELIEF VECTORS"

    addAPBitCorpusAP25K(corpusAPFinal) #add voice bit to belief vectors
    addAPBitCorpusAP25Kto150DSS(corpusAPFinal) #add voice bit to dss150 vectors
    setupEquivalentsAP25K(corpusAPFinal)
    print "Added act/pas BIT TO TRAINITEMS, AND EQUIVALENTS SET"
    
    mapIndexWord=getWordLocalistMap(wordLocalistMapPath)
    mapWordToIndex=getMapWordToIndex(mapIndexWord)
    
    corpusAPFinal=getDerLengthsTrainingVectors(corpusAPFinal,mapWordToIndex)
    print "DERIVATION LENGTHS VECTORS ADDED"
    
    corpusAPFinal.saveToPickle("corpusAPFinal_"+tag+".pick")
    return corpusAPFinal

############################################################################################################
### TEST CONDITIONS 6-9
############################################################################################################
def getCorpusCondition6_8(corpusAP,filename,phrase,phrase2=""):
    '''
    Gets a corpusAP and returns 3 list, one (test items) with the situations that contain a sentence with "phrase" and "phrase2". 
    The second list contains a trainingset withh all TrainingElements of the remaining situations
    The third list contains a test set for training, where only one TrainingElement is given per situation in the training set.
    '''
    cond=[]
    trainingSet=[]
    trainTestSet=[]
    
    def check_validity(elements_situation,phrase,phrase2):
        for elem in elements_situation:
            if elem.testItem.find(phrase) !=-1 and elem.testItem.find(phrase2) !=-1:
                return True
        return False
       
    for situ in corpusAP.actpas:
        if check_validity(situ.actives, phrase,phrase2): cond.append(situ.actives[0])
        else:
            trainTestSet.append(situ.actives[0])
            trainingSet.extend(situ.actives)
        
        if check_validity(situ.passives,phrase,phrase2): cond.append(situ.passives[0])
        else:
            trainTestSet.append(situ.passives[0])
            trainingSet.extend(situ.passives)
        
    for situ in corpusAP.act:
        if check_validity(situ.actives, phrase, phrase2):cond.append(situ.actives[0])
        else:
            trainTestSet.append(situ.actives[0])
            trainingSet.extend(situ.actives)
        
    for tItem in cond:
        print tItem.testItem
        
    print len(trainingSet)
    cond_fold= Fold ([trainingSet,trainTestSet],[cond])
    cond_fold.saveToPickle(filename)
    return cond_fold

def get_imaginary_TrainingElement(tItem,bevents,voice):
    import copy
    all_locations=["bedroom","bathroom","street","playground"]
    all_people=["charlie","sophia","heidi"]
    
    resulting_TEs=[]
    
    if tItem.semantics.find("chess")>-1:game="chess"
    elif tItem.semantics.find("soccer")>-1:game="soccer"
    elif tItem.semantics.find("hide_and_seek")>-1:game="hide_and_seek"
    else:game="nogame"
    
    if tItem.semantics.find("charlie")>-1:person="charlie"
    elif tItem.semantics.find("sophia")>-1:person="sophia"
    elif tItem.semantics.find("heidi")>-1:person="heidi"
    else: person="noperson"
    
    if game=="chess":locations=["bathroom","playground","street"]
    if game=="soccer":locations=["bathroom","playground","bedroom"]
    if game=="hide_and_seek":locations=["street"]
    
    for target_place in locations:
        newTE=copy.deepcopy(tItem)
        if voice=="active":newTE.testItem=person+" plays "+game+" in the "+target_place+" ."
        elif voice=="passive":newTE.testItem=game+" is played by "+person+" in the "+target_place+" ."

        #target_preds=["place("+persona+","+target_place+")" for persona in all_people]
        target_preds=["place("+person+","+target_place+")"]
        anti_target_preds=[]
        for persona in all_people:
            anti_target_preds_p=["place("+persona+","+location+")" for location in all_locations if location != target_place]
            anti_target_preds.extend(anti_target_preds_p)
        
        for targ_pred in target_preds:
            newTE.DSSValue[bevents.index(targ_pred)]=1.0
       
        for anti_target_p in anti_target_preds:
            newTE.DSSValue[bevents.index(anti_target_p)]=0.0
        
        resulting_TEs.append(newTE)
        #===================================================================
        # for ((index,event),condp) in zip(enumerate(bevents),newTE.DSSValue):
        #     print index,event,condp
        #
    return resulting_TEs
            
def getCorpusCondition_Imaginary(corpusAP,bevents,filename):
    '''
    Gets a corpusAP and returns 3 list, one with the situations that are not allowed by the microworld (e.g. charlie plays soccer in the bedroom) 
    The second list contains a trainingset withh all TrainingElements in the dataset
    The third list contains a test set for training, where only one TrainingElement is given per situation in the training set.
    '''
    condA_selected=[]
    condP_selected=[]
    condA=[]
    condP=[]
    trainingSet=[]
    trainTestSet=[]
    
    phrases_active=["plays chess .","plays soccer .",'plays hide_and_seek .']
    phrases_passive=["chess is played by","soccer is played by", "hide_and_seek is played by" ]
       
    for situ in corpusAP.actpas:
        testSitu=False
        for item in situ.actives:
            for phrase in phrases_active:
                if (item.testItem.find(phrase) != -1) and (item.testItem.find("someone")==-1) and (item.testItem.find("a girl")==-1):
                    testSitu=True
                    #print item.testItem
                    break
        if testSitu: condA_selected.append(situ.actives[0])
        trainingSet.extend(situ.actives)
        trainTestSet.append(situ.actives[0])
        
        testSitu=False
        for item in situ.passives:
            for phrase in phrases_passive:
                if (item.testItem.find(phrase) > -1) and (item.testItem.find("someone")==-1) and (item.testItem.find("a girl")==-1)and (item.testItem.find("inside")==-1) and (item.testItem.find("outside")==-1) and (item.testItem.find("in the")==-1):
                    testSitu=True
                    #print item.testItem
                    break
        if testSitu: condP_selected.append(situ.passives[0])
        trainingSet.extend(situ.passives)
        trainTestSet.append(situ.passives[0])
        
    for situ in corpusAP.act:
        trainingSet.extend(situ.actives)
        trainTestSet.append(situ.actives[0])
    
    for tItem in condA_selected:
        print tItem.testItem
        condA.extend(get_imaginary_TrainingElement(tItem, bevents,"active"))
    print 
    for tItem in condA:
        print tItem.testItem
        
    print
    for tItem in condP_selected:
        print tItem.testItem
        condP.extend(get_imaginary_TrainingElement(tItem, bevents,"passive"))
        
    print
    for tItem in condP:
        print tItem.testItem
        
    cond_fold= Fold ([trainingSet,trainTestSet],[condA,condP])
    cond_fold.saveToPickle(filename)
    return cond_fold



#############################################################################################################
#############################################################################################################
#############################################################################################################

if __name__ == '__main__':  

 
 #==============================================================================
 #    ####TO GET THE CURRENT FILES WITH SCHEMA INFO and original 150dss values
 #    activeSentencesPrologFile="dataFiles/rawPrologOutput/model.prod_train.localist_schemas.set"
 #    passiveSentencesPrologFile="dataFiles/rawPrologOutput/model.prod_train_passive.localist_schemas.set"
 #    dssMatrixFile="dataFiles/model.observations"
 #    label="thesisCorpus"
 #    activeSents150dims="dataFiles/rawPrologOutput/filesOriginal150DSS/model.prod_train.localist.set"
 #    passiveSents150dims="dataFiles/rawPrologOutput/filesOriginal150DSS/model.prod_train_passive.localist.set"
 #      
 #    corpusAPFinal=processCorpusAP25KFromRaw(activeSentencesPrologFile,passiveSentencesPrologFile,dssMatrixFile,label,activeSents150dims,passiveSents150dims)
 #         
 # 
 #    corpusAPFinal=CorpusAP()
 #    corpusAPFinal.loadFromPickle("dataFiles/files-thesis/corpusAPFinal_thesis.pick")
 #    
 #    from crossValidation import getKFinalTrainTestCondFolds
 #    getKFinalTrainTestCondFolds(10,corpusAPFinal,"thesis",14,"trainTest_Cond-thesis")
 #     
 #    corpusLMFilename="dataFiles/files-thesis/corpusUID-thesis.pick"
 #    corpusLM=getCorpusUID(corpusAPFinal, corpusLMFilename)
 #    
 #==============================================================================
 
    corpusAPFinal=CorpusAP()
    corpusAPFinal.loadFromPickle("dataFiles/corpusAPFinal.pick")
    _,events=getAtomicEventDSSMap(dsssMatrixPath)
    
    #fold_6=getCorpusCondition6_8(corpusAPFinal, "dataFiles/corpus_folds/fold_cond6_plays_chess.pick", " plays chess")
    #fold_6=getCorpusCondition6_8(corpusAPFinal, "dataFiles/corpus_folds/fold_cond6_plays_soccer.pick", " plays soccer")
    #fold_6=getCorpusCondition6_8(corpusAPFinal "dataFiles/corpus_folds/fold_cond6_plays_hideandseek.pick", " plays hide_and_seek",)
    
    #fold_7=getCorpusCondition6_8(corpusAPFinal, "dataFiles/corpus_folds/fold_cond7_loses_at_chess.pick","loses","at chess")
    #fold_7=getCorpusCondition6_8(corpusAPFinal, "dataFiles/corpus_folds/fold_cond7_loses_at_soccer.pick","loses", " at soccer")
    #fold_7=getCorpusCondition6_8(corpusAPFinal, "dataFiles/corpus_folds/fold_cond7_loses_at_hideandseek.pick","loses"," at hide_and_seek")
    
    #fold_8=getCorpusCondition6_8(corpusAPFinal, "dataFiles/corpus_folds/fold_cond8_chess_is_played.pick", "chess is played")
    #fold_8=getCorpusCondition6_8(corpusAPFinal", "dataFiles/corpus_folds/fold_cond8_soccer_is_played.pick", "soccer is played)
    #fold_8=getCorpusCondition6_8(corpusAPFinal, "dataFiles/corpus_folds/fold_cond8_hideandseek_is_played.pick", "hide_and_seek is played")
    
    fold_cond_imaginary=getCorpusCondition_Imaginary(corpusAPFinal,events,"dataFiles/corpus_folds/fold_cond_imaginary11.pick")
    
    for item in fold_cond_imaginary.valtestSet[0]:
        print item.testItem
        for ev,condp in zip(events,item.DSSValue):
            print ev,condp
        print
        
    #corpusAPFinal.find_and_print_sentence(events,"hide_and_seek is played by charlie .")
                    
                    
        
       


    