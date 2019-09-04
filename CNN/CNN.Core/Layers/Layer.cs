namespace CNN.Core.Layers
{
    using CNN.BL.Enums;
    using CNN.Core.Models;
    using System.Collections.Generic;

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
