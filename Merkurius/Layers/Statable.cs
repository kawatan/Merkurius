namespace Merkurius
{
    namespace Layers
    {
        public interface IStatable
        {
            Batch<double[]> State
            {
                get;
                set;
            }
        }
    }
}
