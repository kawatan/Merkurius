using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using Merkurius;
using Merkurius.Layers;

namespace Seq2seqTest
{
    [DataContract]
    public class Decoder : Layer, IUpdatable, IStatable
    {
        [DataMember]
        private Embedding embedding = null;
        [DataMember]
        private LSTM recurrent = null;
        [DataMember]
        private FullyConnected fullyConnected = null;
        [DataMember]
        private double[] weights = null;

        public double[] Weights
        {
            get
            {
                return this.weights;
            }
            set
            {
                this.weights = value;
            }
        }

        public Batch<double[]> State
        {
            get
            {
                return this.recurrent.State;
            }
            set
            {
                this.recurrent.State = value;
            }
        }

        public Decoder(int sequenceLength, int vocabularySize, int wordVectorSize, int hiddenSize) : base(sequenceLength, sequenceLength * vocabularySize)
        {
            this.embedding = new Embedding(sequenceLength, vocabularySize, wordVectorSize, (fanIn, fanOut) => 0.01 * Initializers.LeCunNormal(fanIn));
            this.recurrent = new LSTM(wordVectorSize, hiddenSize, sequenceLength, true, false, (fanIn, fanOut) => Initializers.LeCunNormal(fanIn));
            this.fullyConnected = new FullyConnected(hiddenSize, sequenceLength, sequenceLength * vocabularySize, (fanIn, fanOut) => Initializers.LeCunNormal(fanIn));
            this.weights = new double[this.embedding.Weights.Length + this.recurrent.Weights.Length + this.fullyConnected.Weights.Length];

            for (int i = 0; i < this.embedding.Weights.Length; i++)
            {
                this.weights[i] = this.embedding.Weights[i];
            }

            for (int i = 0, j = this.embedding.Weights.Length; i < this.recurrent.Weights.Length; i++, j++)
            {
                this.weights[j] = this.recurrent.Weights[i];
            }

            for (int i = 0, j = this.embedding.Weights.Length + this.recurrent.Weights.Length; i < this.fullyConnected.Weights.Length; i++, j++)
            {
                this.weights[j] = this.fullyConnected.Weights[i];
            }
        }
        
        public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
        {
            for (int i = 0; i < this.embedding.Weights.Length; i++)
            {
                this.embedding.Weights[i] = this.weights[i];
            }

            for (int i = 0, j = this.embedding.Weights.Length; i < this.recurrent.Weights.Length; i++, j++)
            {
                this.recurrent.Weights[i] = this.weights[j];
            }

            for (int i = 0, j = this.embedding.Weights.Length + this.recurrent.Weights.Length; i < this.fullyConnected.Weights.Length; i++, j++)
            {
                this.fullyConnected.Weights[i] = this.weights[j];
            }

            return this.fullyConnected.Forward(this.recurrent.Forward(this.embedding.Forward(inputs, isTraining), isTraining), isTraining);
        }

        public override Batch<double[]> Backward(Batch<double[]> deltas)
        {
            return this.embedding.Backward(this.recurrent.Backward(this.fullyConnected.Backward(deltas)));
        }

        public Batch<double[]> GetGradients()
        {
            var vectorList = new List<double[]>();
            var embeddingGradients = this.embedding.GetGradients();
            var recurrentGradients = this.recurrent.GetGradients();
            var fullyConnectedGradients = this.fullyConnected.GetGradients();

            for (int i = 0; i < embeddingGradients.Size; i++)
            {
                vectorList.Add(embeddingGradients[i].Concat<double>(recurrentGradients[i]).Concat<double>(fullyConnectedGradients[i]).ToArray<double>());
            }

            return new Batch<double[]>(vectorList);
        }

        public void SetGradients(Func<bool, double, int, double> func)
        {
            this.embedding.SetGradients(func);
            this.recurrent.SetGradients(func);
            this.fullyConnected.SetGradients(func);
        }

        public void Update(Batch<double[]> gradients, Func<double, double, double> func)
        {
            var vectorList1 = new List<double[]>();
            var vectorList2 = new List<double[]>();
            var vectorList3 = new List<double[]>();

            for (int i = 0; i < gradients.Size; i++)
            {
                var vector1 = new double[this.embedding.Weights.Length];
                var vector2 = new double[gradients[i].Length - this.embedding.Weights.Length - this.fullyConnected.Weights.Length - this.fullyConnected.Outputs];
                var vector3 = new double[this.fullyConnected.Weights.Length + this.fullyConnected.Outputs];

                for (int j = 0; j < this.embedding.Weights.Length; j++)
                {
                    vector1[j] = gradients[i][j];
                }

                for (int j = 0, k = this.embedding.Weights.Length; j < vector2.Length; j++, k++)
                {
                    vector2[j] = gradients[i][k];
                }

                for (int j = 0, k = gradients[i].Length - vector3.Length; j < vector3.Length; j++, k++)
                {
                    vector3[j] = gradients[i][k];
                }

                vectorList1.Add(vector1);
                vectorList2.Add(vector2);
                vectorList3.Add(vector3);
            }

            this.embedding.Update(new Batch<double[]>(vectorList1), func);
            this.recurrent.Update(new Batch<double[]>(vectorList2), func);
            this.fullyConnected.Update(new Batch<double[]>(vectorList3), func);

            for (int i = 0; i < this.embedding.Weights.Length; i++)
            {
                this.weights[i] = this.embedding.Weights[i];
            }

            for (int i = 0, j = this.embedding.Weights.Length; i < this.recurrent.Weights.Length; i++, j++)
            {
                this.weights[j] = this.recurrent.Weights[i];
            }

            for (int i = 0, j = this.embedding.Weights.Length + this.recurrent.Weights.Length; i < this.fullyConnected.Weights.Length; i++, j++)
            {
                this.weights[j] = this.fullyConnected.Weights[i];
            }
        }
    }
}
