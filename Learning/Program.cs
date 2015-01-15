using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Learning
{
    class Program
    {
        // This function reads input from text files
        static List<Matrix> ReadData (string filename)
        {
            List<Matrix> L = new List<Matrix>();

            string line;
            System.IO.StreamReader file = new System.IO.StreamReader(filename);
            while ((line = file.ReadLine()) != null)
            {
                string[] a = line.Split(' ');

                double[,] b = new double[a.Length,1];

                for (int i = 0; i < a.Length; i++)
                {
                    b[i, 0] = Convert.ToDouble(a[i]);
                }

                L.Add(new Matrix(b, a.Length, 1));
            }

            file.Close();

            return L;
        }

        static void Main(string[] args)
        {
            // Read training data
            List<Matrix> X = ReadData("input.txt");
            List<Matrix> Y = ReadData("output.txt");

            // Determinte the number ot nodes in the input and output layer
            int inputLayersCount = X[0].N;
            int outputLayersCount = Y[0].N;

            // Expect input from the user
            Console.WriteLine("Input how many hidden layers the network will have : ");
            int hiddenLayersCount = Convert.ToInt32(Console.ReadLine());
            
            // Input the number of nodes for each hidden layer
            List<int> hiddenLayers = new List<int>();
            for (int i = 0; i < hiddenLayersCount; i++)
            {
                Console.Write("Input node count in hidden layer " + (i + 1) + " : ");
                hiddenLayers.Add(Convert.ToInt32(Console.ReadLine()));
            }

            
            // Holds the number of nodes for each of the hidden layers
            List<int> layers = new List<int>();
            
            layers.Add(X[0].N);
                        
            foreach( int x in hiddenLayers) {
                layers.Add(x);
            }
            
            layers.Add(Y[0].N);

            // Create a new Neural Network            
            NeuralNetwork N = new NeuralNetwork(layers, 0.0005);
          
            // The loop goes on untill the new cost function is an improvement to the old one
            double lastCostFunctionValue;
            do
            {
                lastCostFunctionValue = N.ComputeCostFunction(X, Y);
                N.ComputeDerivatives(X, Y);
                N.Descend();
            } while (lastCostFunctionValue > N.ComputeCostFunction(X, Y));

            int z = 0;

            Console.WriteLine("Cost function value after training is : " + N.ComputeCostFunction(X, Y));

            while ( z != -1)
            {
                Console.WriteLine("Enter new test input vector : ");
                double [,] l = new double[inputLayersCount, 1];

                for (int i = 0; i < inputLayersCount; i++)
                {
                    l[i, 0] = Convert.ToInt32(Console.ReadLine());
                }

                Console.WriteLine("Predicted output is : ");

                N.ForwardPropagate(new Matrix(l, inputLayersCount, 1)).Print();


            }


            Console.Write("\n\nPress any key to continue...");
            Console.ReadLine();
        }
    }
}

