using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Learning
{
    class NeuralNetwork
    {
        public List<int> L;
        
        public List<Matrix> a;
        public List<Matrix> z;

        public List<Matrix> O;

        public List<Matrix> d;
        public List<Matrix> D;

        public double LearningRate;

        public NeuralNetwork(List<int> layers, double learningRate)
        {
            this.L = layers;
            this.LearningRate = learningRate;
            
            this.a = new List<Matrix>();
            this.z = new List<Matrix>();
            this.O = new List<Matrix>();
            this.D = new List<Matrix>();

            foreach (int x in layers)
            {
                a.Add(new Matrix(x, 1, 0));
                z.Add(new Matrix(x, 1, 0));
            }

            for (int i = 0; i < layers.Count - 1; i++)
            {
                Random a = new Random();

                O.Add(new Matrix(layers[i + 1], layers[i], 1));

                O.LastOrDefault().Apply(delegate(double x) { return a.Next(0, 4); });
                
                D.Add(new Matrix(layers[i + 1], layers[i], 1));
            }
        }


        public void Descend()
        {
            List<Matrix> OO = new List<Matrix>();

            for (int k = 0; k < O.Count; k++)
            {
                OO.Add(new Matrix(O[k].N, O[k].M, 0));
            }

            for (int k = 0; k < O.Count; k++)
            {
                for (int i = 0; i < O[k].N; i++)
                {
                    for (int j = 0; j < O[k].M; j++)
                    {
                        OO[k][i, j] = O[k][i, j] - this.LearningRate * D[k][i,j];
                    }
                }
            }

            this.O = OO;          

        }

        
        public void ComputeDerivatives(List<Matrix> X, List<Matrix> Y)
        {
            // Training examples
            int M = X.Count();

            // Default values for D
            for (int i = 0; i < this.D.Count(); i++)
            {
                D[i].Apply(delegate(double x) { return 0; });
            }

            // Computing all the derivatives
            for (int k = 0; k < M; k++)
            {
                // Step 1: Compute all errors d for every node
                List<Matrix> d = this.BackPropagate(X[k], Y[k]);

                // Step 2: Add the errors to the Derivatives
                for (int i = 0; i < this.D.Count; i++)
                {
                    this.D[i] += Matrix.Multiply(d[i + 1], this.a[i].Transpose());
                }
            }

            for (int k = 0; k < D.Count; k++)
            {
                D[k].Apply(delegate(double x) { return x / M; });
            }

        }

        public Matrix ForwardPropagate(Matrix X)
        {
            this.a[0] = new Matrix(X);

            for (int i = 1; i < a.Count; i++)
            {
                this.z[i] = Matrix.Multiply(this.O[i - 1], this.a[i - 1]);
                this.a[i] = this.g(z[i]);
            }

            return this.a.LastOrDefault();
        }

        public List<Matrix> BackPropagate(Matrix X, Matrix Y)
        {
            // 'd' Vectors are the same as 'a' vectors
            List<Matrix>d = new List<Matrix>();

            for (int i = 0; i < this.a.Count - 1; i++)
            {
                d.Add(new Matrix(this.a[i].N, this.a[i].M));
            }

            // The Final Vector in 'd' is h(x)
            d.Add(this.ForwardPropagate(X) - Y);

            // Calculate errors via the forumla d[i] = (O[i]^T) * d[i+1] .* (a[i] .* (1 - a[i]))
            for (int i = d.Count - 2; i >= 1; i--)
            {
                d[i] = Matrix.Multiply(O[i].Transpose(), d[i + 1]) * (this.a[i] * this.a[i].Apply(delegate(double x) { return 1 - x; }));
            }

            return d;
        }

        public double ComputeCostFunction(List<Matrix> X, List<Matrix> Y)
        {
            int M = X.Count();
            double S = 0;

            // Compute the error for each training example
            for (int i = 0; i < M; i++)
            {   
                S += this.computeClassificationCost(X[i], Y[i]);
            }

            S /= (-M);

            return S;
        }


        private double computeRegresionCost(Matrix X, Matrix Y)
        {
            // The formula is (h(x) - y)^2
            return (this.ForwardPropagate(X) - Y).Apply(delegate(double x) { return x * x; }).GetSum();
        }

        private double computeClassificationCost(Matrix X, Matrix Y)
        {
            // The formula is y*log(h(x) + (1-y)(log(1-h(x))
            Matrix a = Y;

            Matrix b = this.ForwardPropagate(X).Apply(delegate(double x) { return Math.Log10(x); });

            Matrix c = Y.Apply(delegate(double y) { return 1 - y; });

            Matrix d = this.ForwardPropagate(X).Apply(delegate(double x) { return Math.Log10(1 - x); });

            return (a * b + c * d).GetSum();
        }


        private Matrix g(Matrix A)
        {
            return A.Apply(delegate(double x) { return 1 / (1 + Math.Pow(Math.E, -x)); });
        }
    }
}
