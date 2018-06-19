#include "network.h"
#include <armadillo>
#include <algorithm>
#include <ctime>
#include <ctype.h>
#include "ctime"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/select.h>
#include <termios.h>

using namespace arma;
using namespace std;



int getch()
{
    int r;
    unsigned char c;
    if ((r = read(0, &c, sizeof(c))) < 0) {
        return r;
    } else {
        return c;
    }
}

int main()
{

    vector<int> hidden; //Vector que incluye  el numero de neuronas por capa oculta
    hidden.push_back(8);hidden.push_back(6);
    Network * my_net = new Network(4,4,hidden,3); // 4 niveles(1 entrada, 2 hidden, 1 salida) ,entrada de 4 parametros,
    my_net->printVector("imprimiendo pesos", my_net->getVectorOrders());
    vector< vector<double >> inputs, outputs, IN;
    int Es=120;// cantidad train
    my_net->loadDataFlowers("irisTraining.txt", Es, inputs, outputs);
    my_net->normalize(IN, inputs);
    //my_net->printMat("Entrada normalizada \n ", IN);
    vector<double> FinalErrors; // error cuadratico medio final
    int times=0;
    bool flag =true;
    double sum;
    while((flag==true) && (times <6000))
    {
        //cout<<"###########################"<< times <<"#################################"""<<endl;
        FinalErrors.clear();
        int era=0;
        double delta=1000;
        for(int i=0 ;i<Es; i++) //Iterar dentro del conjunto de entrenamiento
        {
            double t=0.00001;
            my_net->init(IN[i],outputs[i], t);
            //cout<<"entrada:  "<< i << "   ****  era ***  "<<era<<endl;
            my_net->forward();
            delta=my_net->sumSquareError();
            //cout<<"SumsquareError de la capa:"<<delta<<endl;
            if(delta>0.000001)
                my_net->backpropagation();
            //my_net->forward();
            FinalErrors.push_back(delta);
            era++;
        }
        sum=0;
        for(int qw =0; qw<FinalErrors.size();qw++)
        {
            // cout<<"  -  "<<FinalErrors[qw]<<endl;
            sum+=FinalErrors[qw];
        }
        //cout<<" solo la sumatoria  "<< sum <<" El tamño del vector"<<FinalErrors.size()<<endl;
        sum = sum / FinalErrors.size();
        if(sum < 0.001)
            flag=false;

        times++;
   }
    cout<<"*********acumulado MENOR AL FLAG **** \n "<<sum<<endl;
    cout<<"%%%%%%%%%%%%%%%%%% TEST %%%%%%%%%%%%%%"<<endl;
    int Test =30;
    vector< vector<double >> I, O, NIT;
    my_net->loadDataFlowers("irisTest.txt", Test, I, O);
    my_net->normalize(NIT, I);
    //my_net->printMat("\n Entrada: \n", NIT);
    //my_net->printMat("\n Salidas: \n", O);
    int total_acierto=0;
    for(int i=0 ;i<Test; i++)
    {
         bool f= my_net->testSet(NIT[i],O[i]);//Contar numero de aciertos de el set de  entrenamiento
         if(f)
           total_acierto++;
    }
    cout<<"\n \n Aciertos \n "<<total_acierto<<endl;
}


