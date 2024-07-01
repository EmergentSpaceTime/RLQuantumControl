abstract type AgentParameters end


abstract type AgentNetworks end


abstract type Agent{
    P <: AgentParameters,
    N <: AgentNetworks,
    M <: Memory,
    O,
    D <: AbstractDevice,
    R <: AbstractRNG,
} end
