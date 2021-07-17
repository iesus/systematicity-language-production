import os, numpy, random, sys

import data.loadFiles as loadFiles
from data.crossValidation import Fold
from tools.similarities import levenSimilarity
from tools.plusplus import xplusplus 
import prodSRNN 

sys.path.append("data")

wordLocalistMapPath='data/dataFiles/map_localist_words.txt'
outputsPath="outputs"


def localistToIndices(localistMatrix):
    return [numpy.argmax(localist) for localist in localistMatrix]

def indicesToWords(indices,indexWordMapping):
    return [indexWordMapping[index] for index in indices]

def wordsToIndices(words,wordIndexMapping):
    return[wordIndexMapping[word] for word in words]

def getEquivSentencesIndicesSet(trainElem):
    return [localistToIndices(equivalent.wordsLocalist) for equivalent in trainElem.equivalents]

def getFolders(outputsPath, params):
    """
    Creates the 3 folders where all results/models will be stored
    folderThisRun: folder containing all the files of this particular run, will be contained in folderThisModel which contains all runs of this specific python file
    bestModel: parameters that achieved best performance on the training set
    lastModel: parameters that the model has at the end of training
    """
    #Create folder that contains all the runs for this python file
    currentFolder=outputsPath+"/"+os.path.basename(__file__).split('.')[0]
    folderThisModel=currentFolder+"_outputs"

    if not os.path.exists(folderThisModel): os.mkdir(folderThisModel)
    
    #Create folder for all the files of this specific run
    folderThisRun=folderThisModel+"/output"
    
    folderThisRun+="_"+params['inputType']
    folderThisRun+="_"+str(params['nhidden'])+"h"
    folderThisRun+="_"+str(params['lr'])+"lr"
    folderThisRun+="_"+str(params['nepochs'])+"ep"
    folderThisRun+="_"+params['label']
    
    if not os.path.exists(folderThisRun): os.mkdir(folderThisRun)
    
    #Create folder for plots
    plotsFolder=folderThisRun+"/plots"
    if not os.path.exists(plotsFolder): os.mkdir(plotsFolder)
    
    #Create folders for best and last model parameters
    bestModel=folderThisRun+"/bestModel"
    if not os.path.exists(bestModel): os.mkdir(bestModel)
    lastModel=folderThisRun+"/lastModel"
    if not os.path.exists(lastModel): os.mkdir(lastModel)
    
    return folderThisRun,bestModel,lastModel,plotsFolder

def mostSimilarEquivalentsLevens(trainingElement,modelProduction):
    '''
    Compares the sentence produced by the model with the set of possible sentences related to the DSS,
    obtains the most similar one with its similarity score
    '''
    #Get the possible sentences using word indices
    equivalentsIndices=[localistToIndices(equivalent.wordsLocalist) for equivalent in trainingElement.equivalents]
    #Compare each possible sentence with the sentence the model produced
    similarities=[levenSimilarity(eq,modelProduction) for eq in equivalentsIndices]
    #Get the most similar one
    mostSimilar=numpy.argmax(similarities, 0)
    
    return (similarities[mostSimilar],equivalentsIndices[mostSimilar])

def evaluateSRNN(srnn, outFile, evalSet,mapIndexWord):
    productions_test=srnn.getModelProductions(evalSet)
    
    simgolds=[mostSimilarEquivalentsLevens(sent,pred) for sent,pred in zip(evalSet,productions_test)]
    similarities=[acc for (acc,_) in simgolds]
    golds=[gold for (_,gold) in simgolds]
       
    predWords=[indicesToWords(pred,mapIndexWord) for pred in productions_test]
    labelWords=[indicesToWords(label,mapIndexWord) for label in golds]
    
    printResults(outFile,predWords,labelWords,similarities,evalSet)

def printResults(outFile, predWords,labelWords, similarities, evalSet):
    accuracyGlobal=numpy.sum(similarities)/len(similarities)
    
    perfect=[]
    almostPerfect=[]
    mildlyBad=[]
    worst=[]
    
    def printSubSet(label,setValues,superSize):
        print label
        outFile.write(label+"\n")
        
        for (pw,lw,_) in setValues:
            print pw,lw
            outFile.write(str(pw)+" "+str(lw)+"\n")
            
        print len(setValues)  #number of sentences that fell under this range
        outFile.write(str(len(setValues))+"\n")
        print len(setValues)/float(superSize)#proportion of these sentences with respect to the whole condition
        outFile.write(str(len(setValues)/float(superSize))+"\n\n")
        print 
    
    for pw, lw, acc,item in zip(predWords, labelWords, similarities, evalSet):
            print item.testItem
            print pw,lw
            print acc 
            print
            outFile.write(item.testItem+"\n")
            outFile.write(str(pw)+" "+str(lw)+"\n")
            outFile.write(str(acc)+"\n\n")
            
            if acc==1.0: perfect.append((pw,lw,acc))
            elif acc>=0.8: almostPerfect.append((pw,lw,acc))
            elif acc>=0.5: mildlyBad.append((pw,lw,acc))
            else: worst.append((pw,lw,acc))
            
    printSubSet("PERFECT INSTANCES",perfect,len(evalSet))
    printSubSet("ALMOST PERFECT",almostPerfect,len(evalSet))
    printSubSet("MILDLY BAD", mildlyBad,len(evalSet))
    printSubSet("WORST INSTANCES", worst,len(evalSet))
              
    print   
    print accuracyGlobal
    outFile.write("\n"+str(accuracyGlobal)+"\n")

def testAllFolds_Precision_Recall_Fscore(mapIndexWord):
    '''
    For conditions 1-5:
    when trying to produce all sentences for each DSS
    '''
    import data.setAnalysis as setAnalyzer  
    from decoder import SentenceDecoder
    
    foldsValues=[[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]]]
    goodsSetSizes=[]
    badsSetSizes=[]
     
    fold=Fold() 
    prefixModelsPath="outputs/cond1-5/output_beliefVector_120h_0.24lr_200ep_dots_5cond_25K_"
    prefixFoldsPath="data/dataFiles/corpus_folds/cond1-5/trainTest_Conditions_finalSchemasWithSimilars96_"
    
    
    for x in xrange(10):
        modelPath= prefixModelsPath+str(x)+"/lastModel"
        srnn.load(modelPath)
        
        foldPath=prefixFoldsPath+str(x)+".pick" 
        fold.loadFromPickle(foldPath) #not sure why we need InputObject defined in containers, because of old fold object
        
        testLists=fold.valtestSet
        for tlist in testLists:
            for elem in tlist:
                loadFiles.addPeriods(elem.equivalents,42)#needed because the version of the corpus is old, a new version would not need this line
            loadFiles.setInputType(tlist,s['inputType'])
        
        decod=SentenceDecoder(srnn,mapIndexWord) 
        listasCond=[testLists[0],testLists[1],testLists[2],testLists[3],testLists[5]]
         
        for index in xrange(len(listasCond)): 
            lista=listasCond[index]
            allValues=[[],[],[]]
             
            bads=0  #goods are 70-bad
            for item in lista:
                indicesSet=getEquivSentencesIndicesSet(item) 
                sentencesModel=decod.getNBestPredictedSentencesPerDSS(item,0.12)
                sentencesModelIndices=[sent.indices for sent in sentencesModel]
            
                prec,rec,fscore=setAnalyzer.precisionRecallFScore(indicesSet,sentencesModelIndices)
                
                if fscore<1.0:
                    bads+=1
                    badsSetSizes.append(len(item.equivalents))
                else:
                    goodsSetSizes.append(len(item.equivalents))
                
                allValues[0].append(prec)
                allValues[1].append(rec)
                allValues[2].append(fscore)
                
            avPrec=numpy.mean(allValues[0], axis=0)
            avRec=numpy.mean(allValues[1], axis=0)
            avFsc=numpy.mean(allValues[2], axis=0)
             
            foldsValues[0][index].append(avPrec)
            foldsValues[1][index].append(avRec)
            foldsValues[2][index].append(avFsc)
            foldsValues[3][index].append(14-bads)
        
    condPrec= numpy.mean(foldsValues[0],axis=1)
    condRec= numpy.mean(foldsValues[1],axis=1)
    condFSc= numpy.mean(foldsValues[2],axis=1)
    condPerf=numpy.mean(foldsValues[3],axis=1)/14.0
    
    sd_condPrec= numpy.std(foldsValues[0],axis=1)
    sd_condRec= numpy.std(foldsValues[1],axis=1)
    sd_condFSc= numpy.std(foldsValues[2],axis=1)
    sd_condPerf=numpy.std(foldsValues[3],axis=1)/14.0
    
     
    avPrecA=numpy.mean(condPrec)
    avRecA=numpy.mean(condRec)
    avFscA=numpy.mean(condFSc)
    avPerf=numpy.mean(condPerf)
    
    sd_avPrecA=numpy.std(condPrec)
    sd_avRecA=numpy.std(condRec)
    sd_avFscA=numpy.std(condFSc)
    sd_avPerf=numpy.std(condPerf)
     
    avGoodSetSize=numpy.mean(goodsSetSizes)
    sdGoodSetSize=numpy.std(goodsSetSizes)
    avBadSetSize=numpy.mean(badsSetSizes)
    sdBadSetSize=numpy.std(badsSetSizes)
     
    print "Precision:"
    print condPrec
    print sd_condPrec
    print avPrecA, sd_avPrecA
    
    print "Recall"
    print condRec
    print sd_condRec
    print avRecA, sd_avRecA
     
    print "Fscore"
    print condFSc
    print sd_condFSc
    print avFscA, sd_avFscA
     
    print "Perfect"
    print condPerf
    print sd_condPerf
    print avPerf, sd_avPerf
     
    print goodsSetSizes
    print avGoodSetSize
    print sdGoodSetSize
    print
    print badsSetSizes
    print avBadSetSize
    print sdBadSetSize



def train_model(s):

    #CREATE SRNN AND INITIALIZE VARS
    srnn = prodSRNN.model(
                              inputDimens=s['inputDimension'],
                              hiddenDimens = s['nhidden'],
                              outputDimens= s['vocab_size']
                     )        
    random.seed(s['seed'])
    folderThisRun,bestModel,lastModel,plotsFolder=getFolders(outputsPath,s)

    outputFile= open(folderThisRun+'/output.txt', 'w+')
    best_sim = -numpy.inf
    bestEp=0
    epErrors=[]
    epSimilarities=[]         
    current_lr = s['lr']
    
    for epoch in xrange(s['nepochs']):
        random.shuffle(train)
             
        #TRAIN THIS EPOCH
        errors=srnn.epochTrain(train,current_lr)
        epErrors.append(sum(errors))
        
        predictions_validate=srnn.getModelProductions(validateList,False)#We don't stop on periods, because at the beginning the model may not know that it has to put a period
         
        #Get a list of pairs (sim,mostSimilar) where sim is the similarity of the most similar sentence (mostSimilar) in the gold sentences of the given dss 
        simgolds=[mostSimilarEquivalentsLevens(sent,pred) for sent,pred in zip(validateList,predictions_validate)]
        #Get only the list of similarities
        similarities=[sim for (sim,_) in simgolds]
        similarity=numpy.sum(similarities)/len(similarities)    
        epSimilarities.append(similarity)    
        
        outputLine='Epoch: '+str(epoch)+' lr: '+str(current_lr)+' similarity: '+str(similarity)
        
        if similarity > best_sim:
            srnn.save(bestModel)
            best_sim = similarity
            bestEp=epoch
            lastChange_LR=epoch#just an aux variable that we can change while keeping track of bestEp         
            outputLine='NEW BEST '+outputLine
        
        outputFile.write(outputLine+'\n')
        print outputLine
               
        plt.figure(100000)
        plt.plot(epErrors)
        plt.savefig(folderThisRun+"/errorsTrainEp.png")
                      
        plt.figure(1000000)
        plt.plot(epSimilarities)
        plt.savefig(folderThisRun+"/similarities.png")
            
        # learning rate halves if no improvement in 15 epochs
        if s['decay'] and (epoch-lastChange_LR) >= 15: 
            current_lr *= 0.5
            lastChange_LR=epoch#we have to reset lastChange_LR, otherwise it will halve each epoch until we get an improvement
            
        model_current_epoch=folderThisRun+"/epoch"+str(epoch)
            
        if not os.path.exists(model_current_epoch): os.mkdir(model_current_epoch)
        srnn.save(model_current_epoch)
        srnn.save(lastModel)
        #TRAINING STOPS IF THE LEARNING RATE IS BELOW THRESHOLD OR IF NO IMPROVEMENT DURING 40 EPOCHS
        if current_lr < 1e-3 or (epoch-bestEp)>=40:     
            break  
        
    outputLine='BEST RESULT: epoch '+str(bestEp)+' Similarity: '+str(best_sim)+' with the model '+folderThisRun
    print outputLine
    outputFile.write(outputLine)
    outputFile.close()
    return srnn


def test_model_cond1to5(fold):
    
    #CREATE SRNN AND INITIALIZE VARS
    srnn = prodSRNN.model(
                              inputDimens=s['inputDimension'],
                              hiddenDimens = s['nhidden'],
                              outputDimens= s['vocab_size']
                     )        
    folderThisRun,bestModel,lastModel,plotsFolder=getFolders(outputsPath,s)
    
    srnn.load(lastModel)  
    outFileTrain= open(folderThisRun+'/outputlast_train.txt', 'w+')
    outFileTest= open(folderThisRun+'/outputlast_test.txt', 'w+')
       
    mapIndexWord=loadFiles.getWordLocalistMap(wordLocalistMapPath)
    evaluateSRNN(srnn, outFileTrain, validateList,mapIndexWord)
    outFileTrain.close()
      
    for index in xrange(len(fold.valtestSet)):
        print "\nCONDITION:"+str(index+1)+"\n"
        outFileTest.write("\nCONDITION:"+str(index+1)+"\n")
        evaluateSRNN(srnn, outFileTest, fold.valtestSet[index],mapIndexWord)     
    outFileTest.close()


def printDifferences_in_Sentences_Sets(expectedIndices,producedIndices, producedProbabilities, mapIndexWord,outFile):
    print "Expected:"
    outFile.write("Expected:\n")
    
    for expected in expectedIndices:
        sentence=" ".join([mapIndexWord[index] for index in expected])
        print sentence
        outFile.write(sentence+"\n")
    outFile.write("\n")
    print len(expectedIndices)
    
    sumaModel=0.0
    print "Produced:"
    for sent,prob in zip(producedIndices,producedProbabilities):
        sentWords=" ".join([mapIndexWord[index] for index in sent])
        print str(prob)+"    \t"+str(sentWords)
        sumaModel+=prob
    print len(producedIndices)
    print "sumProdModelPs:"+str(sumaModel)  
    
    print
    for generated in producedIndices:
        if generated not in expectedIndices:
            sentWords=" ".join([mapIndexWord[index] for index in generated])
            outFile.write("\tover:\t"+sentWords+"\n")
            print "overgenerated: "+sentWords
    print
    for expec in expectedIndices:
        if expec not in producedIndices:
            sentWords=" ".join([mapIndexWord[index] for index in expec])
            outFile.write("\t\tunder:\t"+sentWords+"\n")
            print "undergenerated: "+sentWords
            
            

            
            

if __name__ == '__main__':
    
    #gamex="hideandseek"
    #gamex="soccer"
    gamex="chess"
    
    wordx="charlie"
    #wordx="sophia"
    #wordx="heidi"
    
    cond=9
    n_folds=5
    
    Train=False
    #Train=True   
    
    if cond==6:
        corpusFilePath="data/dataFiles/corpus_folds/fold_cond6_plays_"+gamex+".pick"
        label="15_40_cond6_plays_"+gamex+"_"
        
        prefixModelsPath="outputs/cond6-9/output_beliefVector_120h_0.24lr_200ep_15_40_cond6_plays_"+gamex+"_"
        
        if gamex=="hideandseek":gamex="hide_and_seek"
        expected_pattern1="plays "+gamex
        expected_pattern2=""
    
    if cond==7:
        corpusFilePath="data/dataFiles/corpus_folds/fold_cond7_loses_at_"+gamex+".pick"
        label="15_40_cond7_loses_at_"+gamex+"_"
        
        prefixModelsPath="outputs/cond6-9/output_beliefVector_120h_0.24lr_200ep_15_40_cond7_loses_at_"+gamex+"_"
        
        if gamex=="hideandseek":gamex="hide_and_seek"
        expected_pattern1="loses"
        expected_pattern2="at "+gamex
    
    
    if cond==8:
        corpusFilePath="data/dataFiles/corpus_folds/fold_cond8_"+gamex+"_is_played.pick"
        label="15_40_cond8_"+gamex+"_is_played_"
        
        prefixModelsPath="outputs/cond6-9/output_beliefVector_120h_0.24lr_200ep_15_40_cond8_"+gamex+"_is_played_"
        
        if gamex=="hideandseek":gamex="hide_and_seek"
        expected_pattern1=gamex+" is played"
        expected_pattern2=""
        
        
    if cond==9:
        corpusFilePath="data/dataFiles/corpus_folds/fold_cond_imaginary.pick"
        label="15_40_cond11_imaginary_"
        
        prefixModelsPath="outputs/cond6-9/output_beliefVector_120h_0.24lr_200ep_15_40_cond9_imaginary_"
        expected_pattern1="SOMETHING"
        expected_pattern2=""
        
        
    
    if len(sys.argv)>1:
        x=1
        s={
             'lr':float(sys.argv[xplusplus("x")]),      #learning rate
             'decay':int(sys.argv[xplusplus("x")]),     #decay on the learning rate if improvement stops
             'nhidden':int(sys.argv[xplusplus("x")]),   #number of hidden units
             'seed':int(sys.argv[xplusplus("x")]),      #seed for random 
             'nepochs':int(sys.argv[xplusplus("x")]),   #max number of training epochs
             'label':sys.argv[xplusplus("x")],          #label for this run
             'load':int(sys.argv[xplusplus("x")]),      #whether the model is already trained or not
             'inputType':sys.argv[xplusplus("x")],      #dss or sitVector or compVector
             'actpas':sys.argv[xplusplus("x")],         #if the inputs are divided in actpas
             'vocab_size':sys.argv[xplusplus("x")],     #size of the vocabulary
             'inputFile':sys.argv[xplusplus("x")]       #FILE containing the input data
         }
           
    else:
        s = {
         'lr':0.24,                 #learning rate 
         'decay':True,              #decay on the learning rate if improvement stops
         'nhidden':120,             #number of hidden units
         'seed':345,                #seed for random
         'nepochs':200,             #max number of training epochs
         'label':"15_40_cond7_loses_at_chess_x",     #label for this run
         'load':True,               #whether the model is already trained or not
         'inputType':'beliefVector',#dss or sitVector or compVector
         'actpas':True,             #if the inputs are divided in actpas
         'vocab_size':43,
         'inputFile':corpusFilePath   #FILE containing the input data
         }
    
    if s['inputType']=='sitVector' or s['inputType']=='compVector' or s['inputType']=="beliefVector": s['inputDimension']=44
    if s['inputType']=='dss': s['inputDimension']=150
    if s['actpas']:s['inputDimension']=s['inputDimension']+1
    
    
    #LOAD FILES
    fold=Fold()
    fold.loadFromPickle(s['inputFile'])
    
    loadFiles.setInputType(fold.trainSet[0],s['inputType'])
    for tList in fold.valtestSet:
        loadFiles.setInputType(tList,s['inputType'])
        
    train=fold.trainSet[0]
    validateList=fold.trainSet[1]# Traintest is used instead of validation
    
    if cond>0:  
        for elem in train:
            if elem.testItem.find(expected_pattern1)>-1 and elem.testItem.find(expected_pattern2)>-1:
                    print "FOUND IN TRAINING SET!"
                    print elem.testItem
                    exit()    


    if Train:
        for x in xrange(n_folds):
            s['label']=label+str(x)
            train_model(s)
        
    else: 
        
        
        from decoder import SentenceDecoder
        import data.setAnalysis as setAnalyzer  

        #CREATE SRNN AND INITIALIZE VARS
        srnn = prodSRNN.model(inputDimens=s['inputDimension'],hiddenDimens = s['nhidden'],outputDimens= s['vocab_size'])        
        
        mapIndexWord=loadFiles.getWordLocalistMap(wordLocalistMapPath)
        mapWordIndex={word:index for index,word in mapIndexWord.items()}
        
        if cond==0:
            testAllFolds_Precision_Recall_Fscore(mapIndexWord)
       
              
        if cond==9:
            for x in xrange(n_folds):    
                print "FOLD:"+str(x)+"\n"
                modelPath= prefixModelsPath+str(x)+"/lastModel"
                srnn.load(modelPath)
                decod=SentenceDecoder(srnn,mapIndexWord)  
                                       
                for test_cond in fold.valtestSet:
                    for item in test_cond:       
                        print item.testItem                 
        
                        #Get model's predictions 
                        sents_produced=decod.getNBestPredictedSentencesPerDSS(item,0.12)  
                        sentencesModelIndices=[sent.indices for sent in sents_produced] 
                        #Get them in human readable form
                        sentencesWordsSplit=[indicesToWords(sent.indices,mapIndexWord) for sent in sents_produced]
                        sents_produced_StringList=[" ".join(splitSent) for splitSent in sentencesWordsSplit]
                        for sent_p in sents_produced_StringList:
                            print "\t",sent_p
                        print
          
        if cond>5 and cond<9:
    
            def divide_sems_by_location(sem_reps):
                locations=['bedroom','playground','bathroom','street']
                location_divide={location:[] for location in locations}
                location_divide['others']=[]
                
                for sem in sem_reps:
                    founded=False
                    for equiv in sem.equivalents:
                        for location in locations:
                            if equiv.testItem.find(location)>-1:
                                location_divide[location].append(equiv.testItem)
                                founded=True
                                break
                        if founded:break #if any sentence matches a location, stop
                        
                    if not founded:location_divide['others'].append(sem.testItem)
                        
                for key in location_divide.keys():
                    print "\n IN  "+key+" !!"
                    for sent in location_divide[key]:
                        print sent
                    print
                    
                return location_divide
            
            def count_ambiguous(divided_by_location,game):
                unambiguous_locations={"chess":[],"soccer":["street"],"hide_and_seek":["bathroom","playground"]}
                
                n_ambiguous=sum([len(divided_by_location[key]) for key in divided_by_location.keys() if key not in unambiguous_locations[game]])
                n_unambiguous=sum([len(divided_by_location[key]) for key in divided_by_location.keys() if key in unambiguous_locations[game]])
                
                return n_ambiguous,n_unambiguous
                
               
            folds_found=[]
            folds_not_found=[]
            
            folds_amb_percs=[]
            folds_unambs_percs=[]
            folds_found_percs=[]
            
            for x in xrange(n_folds):    
                
                fold_found=[]
                fold_not_found=[]   
                    
                modelPath= prefixModelsPath+str(x)+"/lastModel"
                srnn.load(modelPath)
                decod=SentenceDecoder(srnn,mapIndexWord)  
                                       
                #IN THESE CONDITIONS THERE IS ONLY ONE TEST LIST
                for item in fold.valtestSet[0]:                     
                    #Get model's predictions 
                    sents_produced=decod.getNBestPredictedSentencesPerDSS(item,0.12)  
                    sentencesModelIndices=[sent.indices for sent in sents_produced] 
                    #Get them in human readable form
                    sentencesWordsSplit=[indicesToWords(sent.indices,mapIndexWord) for sent in sents_produced]
                    sents_produced_StringList=[" ".join(splitSent) for splitSent in sentencesWordsSplit]
                    
                    #Get expected sentences 
                    sentsExpected=[equiv.testItem for equiv in item.equivalents]
                    indicesSet=getEquivSentencesIndicesSet(item)
                        
                    prec,rec,fscore=setAnalyzer.precisionRecallFScore(indicesSet,sentencesModelIndices)
                    
                    correct_productions=[sent_produced for sent_produced in sents_produced_StringList if sent_produced in sentsExpected]
                    incorrect_productions=[sent_produced for sent_produced in sents_produced_StringList if sent_produced not in sentsExpected]
                    
                    pattern_found=False
                    for prod in correct_productions:
                        if prod.find(expected_pattern1)>-1 and prod.find(expected_pattern2)>-1:
                            pattern_found=True
                            break
    
                    if pattern_found:
                        print x, item.testItem
                        fold_found.append((x,item,sents_produced_StringList))
                    else: fold_not_found.append((x,item,sents_produced_StringList))
                
                folds_found.append(fold_found)
                folds_not_found.append(fold_not_found)
                
                sems_in_found=[sem for (f,sem,sents_produced) in fold_found]
                div_found=divide_sems_by_location(sems_in_found)
                div_all=divide_sems_by_location(fold.valtestSet[0])
                n_total=len(fold.valtestSet[0])
                print n_total
               
                n_amb_all,n_unamb_all=count_ambiguous(div_all, gamex)
                n_amb_found,n_unamb_found= count_ambiguous(div_found,gamex)
                
                print n_amb_all,n_unamb_all
                print n_amb_found,n_unamb_found
                if n_amb_all !=0:
                    amb_perc=(n_amb_found*1.0/n_amb_all)*100.0
                    print "ambiguous perc found: ",amb_perc
                    folds_amb_percs.append(amb_perc)
                if n_unamb_all!=0:
                    unamb_perc=(n_unamb_found*1.0/n_unamb_all)*100.0
                    print "unambiguous perc found: ",unamb_perc
                    folds_unambs_percs.append(unamb_perc)
                    
                folds_found_percs.append((len(fold_found)*1.0/n_total)*100)
                
            print folds_amb_percs
            print folds_unambs_percs
            print folds_found_percs
            
            print sum(folds_amb_percs)/10
            print sum(folds_unambs_percs)/10
            print sum(folds_found_percs)/10
            
            print numpy.std(folds_amb_percs)
            print numpy.std(folds_unambs_percs)
            print numpy.std(folds_found_percs)
            
            
            
                
           
            
        
        


