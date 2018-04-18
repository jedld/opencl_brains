module TensorStream
  module Train
    class GradientDescentOptimizer
      attr_accessor :learning_rate

      def initialize(learning_rate, options = {})
        @learning_rate = learning_rate
      end

      def minimize(cost)
        trainable_vars = TensorStream::Graph.get_default_graph.get_collection(GraphKeys::GLOBAL_VARIABLES).select(&:trainable?)
        operations = []

        trainable_vars.collect do |v|
          Operation.derivative(cost.eval(retain: [v]))
        end
        operations
      end
    end
  end
end