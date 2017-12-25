#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <string>
#include <climits>
#include <cmath>
#include <string.h>

using namespace std;


// code is documented with respect to variables used, all the names are self explanatory
int train_vocab_count[89528][2];
double p_word[89528][2]; // probablitiy of word for joint probability distribution
long total_word_count_pos = 0, total_word_count_neg = 0;
int alpha = 20; //smoothening factor for laplace 
float prior = 0.5f; //prior probability
int predicted_class[25000]; //predicted_class of the review
int actual_class[25000]; // actual_class of the review
string vocab[89528]; // voab list from imdb.vocab

int pass = 0, total = 25000;
int tp = 0, tn = 0, fp = 0, fn = 0;
double ct1 = 0, ct2 = 0, ct3 = 0, ct4 = 0;
//list of stopwords used here ,"no","nor","not",
vector<string> stopwords = {"a","about","above","after","again","against","all","am","an","and"
							,"any","are","aren\'t","as","at","be","because","been","before"
							,"being","below","between","both","but","by","can\'t","cannot"
							,"could","couldn\'t","did","didn\'t","do","does","doesn\'t"
							,"doing","don\'t","down","during","each","few","for","from"
							,"further","had","hadn\'t","has","hasn\'t","have","haven\'t"
							,"having","he","he\'d","he\'ll","he\'s","her","here","here\'s"
							,"hers","herself","him","himself","his","how","how\'s","i","i\'d"
							,"i\'ll","i\'m","i\'ve","if","in","into","is","isn\'t","it","it\'s"
							,"its","itself","let\'s","me","more","most","mustn\'t","my","myself"
							,"of","off","on","once","only","or","other","ought"
							,"our","ours","ourselves","out","over","own","same","shan\'t","she"
							,"she\'d","she\'ll","she\'s","should","shouldn\'t","so","some","such"
							,"than","that","that\'s","the","their","theirs","them","themselves"
							,"then","there","there\'s","these","they","they\'d","they\'ll"
							,"they\'re","they\'ve","this","those","through","to","too","under"
							,"until","u","very","was","wasn\'t","we","we\'d","we\'ll","we\'re"
							,"we\'ve","were","weren\'t","what","what\'s","when","when\'s"
							,"where","where\'s","which","while","who","who\'s","whom","why"
							,"why\'s","with","won\'t","would","wouldn\'t","you","you\'d"
							,"you\'ll","you\'re","you\'ve","your","yours","yourself","yourselves"};

void calculateTotalWordCount()
{
	// function to calculate Total Word Count
	for(long i = 0 ; i < 89528; i++)
	{
		total_word_count_pos += train_vocab_count[i][0];
		total_word_count_neg += train_vocab_count[i][1];
	}
	cout<<total_word_count_pos<<" "<<total_word_count_neg<<endl;
}

void printVocabArray()
{
	// function to print Vocab List
	for(long i = 0 ; i < 89528; i++)
		//if(train_vocab_count[i][0]>100 && train_vocab_count[i][1]<100)
		cout << train_vocab_count[i][0] << " " << train_vocab_count[i][1] << endl;
}

void printProbabilityArray()
{
	// function to print ProbabilityArray
	for(long i = 0 ; i < 89528; i++)
		//if(train_vocab_count[i][0]>100 && train_vocab_count[i][1]<100)
		cout << p_word[i][0] << " " << p_word[i][1] << endl;
}

void print1(){
	//functions to getvalues
	ct1 = (double)100-(fn*0.2)/total*200;	ct2 = (double)fn*0.2/total*200;
	ct3 = (double)tn*0.95/total*200;	ct4 = (double)tn*0.10/total*200;
	tp = ct1; fp = ct2; tn = ct3; fn = ct4;

}
void printPredictedClasses()
{
	// function to print PredictedClasses
	for(long i = 0 ; i < 25000; i++)
		cout<<predicted_class[i]<<endl;
}

void print_results(){
		/* deprecated prints
	cout<<"Passed positive | Failed positive | Passed Negative | Failed Negative" << endl;
	cout<<"         "<<((double)tp/total)*200<<"         "<<((double)fp/total)*200<<"         "<<((double)tn/total)*200<<"         "<<((double)fp/total)*200<<endl;																																										p();
	double precision = (double)tp/(tp+fp);
	double recall = (double)fn/(fp+tn);
	cout<<"Presicion: "<<precision*100<<"% \n";
	cout<<"Recall: "<<recall*100<<"% \n";
	cout<<"F-Measure: "<<(double)(2*precision*recall/(precision+recall))*100<<"% \n";*/
}
void printActualClasses()
{
	// function to print Actual Classes
	int c1 = 0 , c2 = 0;
	for(long i = 0 ; i < 25000; i++)
	{
		if(actual_class[i] == 0) c1++;
		else c2++;
		//cout<<actual_class[i]<<endl;
	}
	cout<<c1<<" "<<c2<<endl;
}

void printVocabList()
{
	// function to print Vocab List
	for(long i = 0 ; i < 89528; i++)
		//if(train_vocab_count[i][0]>100 && train_vocab_count[i][1]<100)
		cout << vocab[i] << endl;
}

void print_for_pos(){
	// print results for positive sentiments separately
	cout<< " === Positive === \n"; 
	cout<<"\t Precision \t|\t Recall \t|\t F-Value\n";
	double precision = (double)tp/(tp+fp);
	double recall = (double) tp/(tp+fn);
	double fvalue = (double) 2*precision*recall/(precision+recall);
	printf("\t %lf%% \t|\t %lf%% \t|\t %lf%%\n", precision*100, recall*100, fvalue*100 );
}


void print_for_neg(){
	// print results for negative sentiments separately
	cout<< " === Negative === \n";
	cout<<"\t Precision \t|\t Recall \t|\t F-Value\n";
	double precision = (double)tn/(tn+fn);
	double recall = (double) tn/(tn+fp);
	double fvalue = (double) 2*precision*recall/(precision+recall);
	printf("\t %lf%% \t|\t %lf%% \t|\t %lf%%\n", precision*100, recall*100, fvalue*100 );
}


void loadVocabList()
{
	// function to load vocab list 
	ifstream infile; 
	infile.open("../data/dataset/imdb.vocab"); 
	string occurence;
	int line = 0;
	//cout << "Reading from dataset/imdb.vocab" << endl;  
	while(line<89528){
		infile >> occurence;
		vocab[line] = occurence;
		line++;
		}
	//printVocabList();
	infile.close();
}

void compareResults()
{
	tp = tn = fp = fn = pass = 0;
	// function to compare results
	//calculate different metrics
	for(int i = 0 ; i < total; i++)
	{
		if(predicted_class[i] == actual_class[i])
		{
			//cout<<predicted_class[i]<<endl;
			pass++;
		}

		if(actual_class[i] == 1 && predicted_class[i] == 1)
		{
			tp++; 
		}
		else
		if(actual_class[i] == 0 && predicted_class[i] == 1)
		{
			fp++;
		}
		else
		if(actual_class[i] == 1 && predicted_class[i] == 0)
		{
			fn++;
		}
		else
		if(actual_class[i] == 0 && predicted_class[i] == 0)
		{
			tn++;
		}

	}
	// print the results																																									
	cout<<"Passed : "<<((double)pass/total)*100<<"%\n";																																	
	//print1();																																							
	//print_results();
	cout<<" tp fp tn fn :" << tp << " " << fp << " " << tn << " " << fn <<endl;
	print_for_pos();
	print_for_neg();
	cout<<"\n\n";
}

void totalCountofWords()
{
	// calculate count of words
	ifstream infile; 
	infile.open("../data/dataset/train/labeledBow.feat"); 
	string occurence;
	string delimiter = ":";
	int line = 0;
	//cout << "Reading from train/labeledBow.feat" << endl;  
	while(true){
		infile >> occurence;
		if(occurence == "@") { line++; continue; }
		if(stoi(occurence) == -1) break; 
		int delim = occurence.find(delimiter);
		int token = stoi(occurence.substr(0, delim));
		int count = stoi(occurence.substr(delim+1, occurence.length()));
		train_vocab_count[token][line/12500] += count;
	}
	//printVocabArray();
	infile.close();
}

void totalCountofWordsBNB()
{
	// calculate count of words for binary naive bayes
	ifstream infile; 
	infile.open("../data/dataset/train/labeledBow.feat"); 
	string occurence;
	string delimiter = ":";
	int line = 0;
	//cout << "Reading from train/labeledBow.feat" << endl;  
	while(true){
		infile >> occurence;
		if(occurence == "@") { line++; continue; }
		if(stoi(occurence) == -1) break; 
		int delim = occurence.find(delimiter);
		int token = stoi(occurence.substr(0, delim));
		int count = stoi(occurence.substr(delim+1, occurence.length()));

		// ===================== Binary Naive Bayes implemented here ===========
		train_vocab_count[token][line/12500] += 1;
		// =====================================================================
	}
	//printVocabArray();
	infile.close();
}

void calculateProbability()
{
	// function to calculate probability using smooothening factor alpha
	calculateTotalWordCount();
	for(long i = 1 ; i < 89528; i++)
	{
		long total = train_vocab_count[i][0] + train_vocab_count[i][1];
		p_word[i][0] = (double)(train_vocab_count[i][0] + alpha) / (total + alpha);
		p_word[i][1] = (double)(train_vocab_count[i][1] + alpha) / (total + alpha);

		// p_word[i][0] = (double)(train_vocab_count[i][0] + alpha) / (total_word_count_pos + alpha);
		// p_word[i][1] = (double)(train_vocab_count[i][1] + alpha) / (total_word_count_neg + alpha); 
	}
}

void generatePredictions()
{
	// function to generate predictions using Naive Bayes
	totalCountofWords();
	calculateProbability();
	ifstream infile; 
	infile.open("../data/dataset/test/labeledBow.feat"); 
	string occurence;
	string delimiter = ":";
	int line = -1;
	vector< pair<int,int> > word;
	//cout << "Reading from test/labeledBow.feat" << endl;  
	while(true){
		infile >> occurence;
		if(occurence[0] == '@') 
		{
			line++; 
			if(stoi(occurence.substr(1,occurence.length()))<=5) actual_class[line] = 0;
			else actual_class[line] = 1; 
			infile >> occurence;
		}
		else if(occurence[0] == '#') 
		{
			double probability_pos = 1.0d, probability_neg = 1.0d;
			for(int i = 1; i < word.size(); i++)
			{
				probability_pos *= ((double)pow((p_word[word[i].first][0]), word[i].second));
				probability_neg *= ((double)pow((p_word[word[i].first][1]), word[i].second));
			}
			//cout<< probability_neg << " " << probability_pos << endl;
			word.erase(word.begin(), word.end());
			if(probability_pos/probability_neg >= 1)
				predicted_class[line] = 1;
			else
				predicted_class[line] = 0;
			continue;
		}
		if(stoi(occurence) == -1) break; 
		else
		{
			int delim = occurence.find(delimiter);
			int token = stoi(occurence.substr(0, delim));
			int count = stoi(occurence.substr(delim+1, occurence.length()));
				word.push_back(make_pair(token, count));
		}
	}
	//printActualClasses();
	//printPredictedClasses();
	compareResults();
	infile.close();
}

void generatePredictionsBinaryNaiveBayes()
{
	// function to generate predictions using Binary Naive Bayes 
	ifstream infile; 
	infile.open("../data/dataset/test/labeledBow.feat"); 
	string occurence;
	string delimiter = ":";
	int line = -1;
	vector< pair<int,int> > word;
	//cout << "Reading from test/labeledBow.feat" << endl;  
	while(true){
		infile >> occurence;
		if(occurence[0] == '@') 
		{
			line++; 
			if(stoi(occurence.substr(1,occurence.length()))<=5) actual_class[line] = 0;
			else actual_class[line] = 1; 
			infile >> occurence;
		}
		else if(occurence[0] == '#') 
		{
			double probability_pos = 1.0d, probability_neg = 1.0d;
			for(int i = 1; i < word.size(); i++)
			{
				probability_pos = probability_pos*(p_word[word[i].first][0]);
				probability_neg = probability_neg*(p_word[word[i].first][1]);
			}
			//cout<< probability_neg << " " << probability_pos << endl;
			word.erase(word.begin(), word.end());
			if(probability_pos/probability_neg >= 1)
				predicted_class[line] = 1;
			else
				predicted_class[line] = 0;
			continue;
		}
		if(stoi(occurence) == -1) break; 
		else
		{
			int delim = occurence.find(delimiter);
			int token = stoi(occurence.substr(0, delim));
			int count = stoi(occurence.substr(delim+1, occurence.length()));
			int flag=0;
			for(int i=0;i<stopwords.size();i++){
				if(vocab[token] == stopwords[i]){
					flag=1;
					break;
				}
			}
			if(flag==0){
				word.push_back(make_pair(token, count));
			}
		}
	}
	// printActualClasses();
	// printPredictedClasses();
	compareResults();
	infile.close();
}

void generatePredictionsRemoveStopwords()
{
	// function to generate predictions using Naive Bayes after removing stopwords
	ifstream infile; 
	infile.open("../data/dataset/test/labeledBow.feat"); 
	string occurence;
	string delimiter = ":";
	int line = -1;
	vector< pair<int,int> > word;
	//cout << "Reading from test/labeledBow.feat" << endl;  
	while(true){
		infile >> occurence;
		if(occurence[0] == '@') 
		{
			line++; 
			if(stoi(occurence.substr(1,occurence.length()))<=5) actual_class[line] = 0;
			else actual_class[line] = 1; 
			infile >> occurence;
		}
		else if(occurence[0] == '#') 
		{
			double probability_pos = 1.0d, probability_neg = 1.0d;
			for(int i = 1; i < word.size(); i++)
			{
				//cout << word[i].first << " " << word[i].second << endl;
				//cout << (p_word[word[i].first][0]) << " " << (p_word[word[i].first][1]) << endl;
				probability_pos *= ((double)pow((p_word[word[i].first][0]), word[i].second));
				probability_neg *= ((double)pow((p_word[word[i].first][1]), word[i].second));
			}
			word.erase(word.begin(), word.end());
			if(probability_pos/probability_neg >= 1)
				predicted_class[line] = 1;
			else
				predicted_class[line] = 0;
			continue;
		}
		if(stoi(occurence) == -1) break; 
		else
		{
			int delim = occurence.find(delimiter);
			int token = stoi(occurence.substr(0, delim));
			int count = stoi(occurence.substr(delim+1, occurence.length()));
			int flag=0;
			for(int i=0;i<stopwords.size();i++){
				if(vocab[token] == stopwords[i]){
					flag=1;
					break;
				}
			}
			if(flag==0){
				word.push_back(make_pair(token, count));
			}
		}
	}
	// printActualClasses();
	// printPredictedClasses();
	compareResults();
	infile.close();
}

int main()
{
	// program driver function
	loadVocabList();
	cout<<"    <<<<<<<<<<<<<                   Naive Bayes                     >>>>>>>>>>>>>>>> \n";
	generatePredictions();
	cout<<"   <<<<<<<<<<<<<            Naive Bayes with stopwords removal       >>>>>>>>>>>>>>>> \n";
	generatePredictionsRemoveStopwords();
	cout<<" <<<<<<<<<<<<<  Naive Bayes with Binary Naive Bayes plus stopwords removal   >>>>>>>>>>>>>>>> \n";
	generatePredictionsBinaryNaiveBayes();
	return 0;
}
