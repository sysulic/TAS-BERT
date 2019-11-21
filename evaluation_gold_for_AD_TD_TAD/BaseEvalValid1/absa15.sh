#!/bin/bash
clear

# Baselines folder (full path)
dir=$(pwd)/

echo "Reading config...." >&2
. $dir"absa15.conf"

# ttd full path
pth=$dir$ttd/


# ---------------------------------------------------

ABSABaseAndEval ()
{

echo -e "***** Harvesting Features from Train *****"
java -cp ./A.jar absa15.Do ExtractFeats $dom $dir $ttd $ftr$pIdxArg

echo -e "***** Creating Train Vectors for Stages 1 and 2 *****"
java -cp ./A.jar absa15.Do CreateVecs $dom $dir $ttd 1 "1"$pIdxArg

echo -e "***** Training SVM model for category  prediction *****"
./libsvm-3.18/svm-train -t 0 -b 1 -q ${pth}"tr.svm.asp"${suff} ${pth}"tr.svm.model.asp"${suff} 

# Stage 1 Predict

echo -e "***** Creating Test Vectors for Stage 1 *****"
java -cp ./A.jar absa15.Do CreateVecs $dom $dir $ttd 2 "1"$pIdxArg

echo -e "***** Predicting categories *****"
./libsvm-3.18/svm-predict -b 1 ${pth}"te.svm.asp"${suff} ${pth}"tr.svm.model.asp"${suff} ${pth}"Out.asp"${suff}

echo -e "***** Assigning categories using a threshold on the SVM prediction *****"
java -cp ./A.jar absa15.Do Assign $dom $dir $ttd $thr 1 "0"$pIdxArg

if [ "$dom" = "rest" ]; then

echo -e "***** Determining targets using a target list created from train data *****"
java -cp ./A.jar absa15.Do IdentifyTargets $dom $dir $ttd $pIdxArg
fi

# Stage 2 Training

echo -e "***** Training polarity category model *****"
./libsvm-3.18/svm-train -t 0 -b 1 -q ${pth}"tr.svm.pol"${suff} ${pth}"tr.svm.model.pol"${suff}

# Stage 2 Predict

echo -e "***** Creating Test Vectors for Stage 2 *****"
java -cp ./A.jar absa15.Do CreateVecs $dom $dir $ttd 2 "2"$pIdxArg

# gold aspects 
echo -e "***** Predicting polarities using SVM for gold aspect categories *****"
./libsvm-3.18/svm-predict -b 1 ${pth}"te.svm.pol4g"${suff} ${pth}"tr.svm.model.pol"${suff} ${pth}"Out.pol"${suff}

echo -e "***** Assigning polarities based on SVM prediction *****"
java -cp ./A.jar absa15.Do Assign $dom $dir $ttd 0 2 "0"$pIdxArg

# pred aspects 
#echo -e "***** Predicting polarities using SVM for predicted aspect categories *****"
#./libsvm-3.18/svm-predict -b 1 ${pth}"te.svm.pol4p"${suff} ${pth}"tr.svm.model.pol"${suff} ${pth}"Out.pol"${suff}

#echo -e "***** Assigning polarities based on SVM prediction *****"
#java -cp ./A.jar absa15.Do Assign $dom $dir $ttd 0 2 "1"$pIdxArg

# Evaluate results

echo -e "\n"
echo -e "***** Evaluate Stage 1 Output (target and category) *****"

java -cp ./A.jar absa15.Do Eval ${pth}"teCln.PrdAspTrg.xml"${suff} ${pth}"teGld.xml"${suff} 1 0 

if [ "$dom" = "rest" ]; then
java -cp ./A.jar absa15.Do Eval ${pth}"teCln.PrdAspTrg.xml"${suff} ${pth}"teGld.xml"${suff} 2 0
java -cp ./A.jar absa15.Do Eval ${pth}"teCln.PrdAspTrg.xml"${suff} ${pth}"teGld.xml"${suff} 3 0
fi

echo -e "***** Evaluate Stage 2 Output (Polarity) *****"
java -cp ./A.jar absa15.Do Eval ${pth}"teGldAspTrg.PrdPol.xml"${suff} ${pth}"teGld.xml"${suff} 5 1 

}

echo -e "*******************************************"
echo -e "BASELINES DIR:" $dir
echo -e "Stage 1: Aspect and OTE extraction"
echo -e "Stage 2: Polarity classification"

echo -e "***** Validate Input XML *****"
java -cp ./A.jar absa15.Do Validate ${dir}${src} ${dir}"ABSA15.xsd" $dom

if [ "$xva" -eq 0 ]; then
	echo -e "***** Split Train Test *****"	
	java -cp ./A.jar absa15.Do Split $sfl $dir $ttd $src $fld $partIdx
	ABSABaseAndEval 
else 
	echo -e "***** Split *****" 	
	java -cp ./A.jar absa15.Do Split $sfl $dir $ttd $src $fld
	echo -e "\n***** Cross Validation*****\n"
	for i in $(eval echo {1..$fld}); do	  		  
	  echo -e "Round " $i	  	  	  
	  pIdxArg=" "$(($i-1))
	  suff="."$(($i-1))
	  echo $pIdxArg $suff
	  ABSABaseAndEval
	  echo -e "\n"
	done
fi
echo -e "*******************************************"


