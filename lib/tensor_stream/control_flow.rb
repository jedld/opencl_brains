module TensorStream
  class ControlFlow < Operation
    attr_accessor :ops

    def initialize(flow_type, items, ops = nil, options = {})
      @operation = :"flow_#{flow_type}"
      @items = items
      @name = set_name
      @ops = ops

      @graph = options[:graph] || TensorStream.get_default_graph
      @graph.add_node(self)
    end

    def run
      eval
    end
  end
end