using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Learning
{
    class GradientDescent
    {

        double LearningRate;
        int MaxAttempts;
        NeuralNetwork NeuNetwork;
        List<Matrix> X, Y;


        public GradientDescent(NeuralNetwork N, List<Matrix> X, List<Matrix> Y, double learningRate, int maxAttemps = 10000)
        {
            this.NeuNetwork = N;
            this.LearningRate = learningRate;
            this.MaxAttempts = maxAttemps;

            this.X = X;
            this.Y = Y;
        }

        public void Descend()
        {
            for (int t = 0; t < this.MaxAttempts; t++)
            {
                Console.WriteLine(NeuNetwork.ComputeCostFunction(this.X, this.Y));

                NeuNetwork.ComputeDerivatives(this.X, this.Y);

                List<Matrix> newO = new List<Matrix>();

                for (int k = 0; k < this.NeuNetwork.O.Count; k++)
                {
                    newO.Add(new Matrix(NeuNetwork.O[k].N, NeuNetwork.O[k].M, 0));
                }

                for (int k = 0; k < NeuNetwork.O.Count; k++)
                {
                    for (int i = 0; i < NeuNetwork.O[k].N; i++)
                    {
                        for (int j = 0; j < NeuNetwork.O[k].M; j++)
                        {
                            newO[k][i, j] = NeuNetwork.O[k][i, j] - this.LearningRate * NeuNetwork.D[k][i, j];
                        }
                    }
                }

                NeuNetwork.O = newO;                
            }
        }
    }
}
