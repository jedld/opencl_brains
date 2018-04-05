module TensorStream
  class TensorShape
    def initialize(shape, rank)
      @shape = shape
      @rank = rank
    end

    def to_s
      dimensions = @shape.collect do |r|
        "Dimension(#{r})"
      end.join(',')
      "TensorShape([#{dimensions}])"
    end
  end
end