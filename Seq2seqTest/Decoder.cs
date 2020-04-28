using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Serialization;
using Merkurius;
using Merkurius.Layers;

namespace Seq2seqTest
{
    [DataContract]
    public class Decoder
    {
        private Embedding embedding = null;
        private LSTM lstm = null;
        private FullyConnected fullyConnected = null;
        
        public Decoder(int sequenceLength, int vocabularySize, int wordVectorSize, int hiddenSize = 256)
        {
            this.embedding = new Embedding(sequenceLength, vocabularySize, wordVectorSize, (fanIn, fanOut) => 0.01 * Initializers.LeCunNormal(fanIn));
            this.lstm = new LSTM(wordVectorSize, hiddenSize, sequenceLength, true, false, (fanIn, fanOut) => Initializers.LeCunNormal(fanIn));
            this.fullyConnected = new FullyConnected(hiddenSize, sequenceLength, sequenceLength * vocabularySize, (fanIn, fanOut) => Initializers.LeCunNormal(fanIn));
        }
    }
}
