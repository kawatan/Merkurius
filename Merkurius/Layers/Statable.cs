namespace Merkurius
{
    namespace Layers
    {
        interface IStatable
        {
            Batch<double[]> State
            {
                get;
                set;
            }
        }
    }
}
