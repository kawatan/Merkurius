using System;
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Text;
using Merkurius;
using Merkurius.ActivationFunctions;
using Merkurius.Layers;

namespace Seq2seqTest
{
    [DataContract]
    public class Seq2seq : Layer, IUpdatable
    {
        public double[] Weights { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public Seq2seq() : base(0, 0)
        {

        }

        public override Batch<double[]> Forward(Batch<double[]> inputs, bool isTraining)
        {
            throw new NotImplementedException();
        }

        public override Batch<double[]> Backward(Batch<double[]> deltas)
        {
            throw new NotImplementedException();
        }

        public Batch<double[]> GetGradients()
        {
            throw new NotImplementedException();
        }

        public void SetGradients(Func<bool, double, int, double> func)
        {
            throw new NotImplementedException();
        }

        public void Update(Batch<double[]> gradients, Func<double, double, double> func)
        {
            throw new NotImplementedException();
        }
    }
}
