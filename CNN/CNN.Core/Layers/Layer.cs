namespace CNN.Core.Layers
{
    using CNN.BL.Enums;

    /// <summary>
    /// Слой нейронной сети.
    /// </summary>
    public abstract class Layer
    {
        /// <summary>
        /// Тип слоя.
        /// </summary>
        public abstract LayerType LayerType { get; }
    }
}
