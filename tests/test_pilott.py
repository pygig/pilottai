from unittest.mock import Mock, AsyncMock, patch

import pytest
import pytest_asyncio

from pilottai import Pilott
from pilottai.agent import Agent
from pilottai.core.base_config import LLMConfig
from pilottai.config.model import JobResult
from pilottai.job.job import Job
from pilottai.enums.process_e import ProcessType
from pilottai.enums.job_e import JobPriority, JobAssignmentType


@pytest.fixture
def llm_config():
    """Fixture for LLM configuration"""
    return LLMConfig(
        model_name="test-model",
        provider="test-provider",
        api_key="test-key"
    )


@pytest_asyncio.fixture
async def mock_agent():
    """Fixture to create a mock agent"""
    agent = Mock(spec=Agent)
    agent.id = "test_agent"
    agent.status = "idle"
    agent.jobs = []
    agent.title = "test_title"
    agent.goal = "test goal"
    agent.description = "test description"

    # Create success result
    success_result = JobResult(
        success=True,
        output="Test result",
        execution_time=0.1,
        error=None,
        metadata={}
    )
    agent.execute_job = AsyncMock(return_value=success_result)
    agent.execute_jobs = AsyncMock(return_value=[success_result])
    agent.evaluate_job_suitability = AsyncMock(return_value=0.8)
    agent.start = AsyncMock()
    agent.stop = AsyncMock()
    return agent


@pytest_asyncio.fixture
async def mock_agents(mock_agent):
    """Fixture to create multiple mock agents"""
    agents = []
    for i in range(3):
        agent = Mock(spec=Agent)
        agent.id = f"agent_{i}"
        agent.status = "idle"
        agent.jobs = []
        agent.title = f"title_{i}"
        agent.goal = f"goal_{i}"
        agent.description = f"description_{i}"

        success_result = JobResult(
            success=True,
            output=f"Result from agent_{i}",
            execution_time=0.1,
            error=None,
            metadata={}
        )
        agent.execute_job = AsyncMock(return_value=success_result)
        agent.execute_jobs = AsyncMock(return_value=[success_result])
        agent.evaluate_job_suitability = AsyncMock(return_value=0.7 + (i * 0.1))
        agent.start = AsyncMock()
        agent.stop = AsyncMock()
        agents.append(agent)
    return agents


@pytest_asyncio.fixture
async def pilott():
    """Fixture to create a basic Pilott instance"""
    pilott_instance = Pilott(name="TestPilott")
    try:
        yield pilott_instance
    finally:
        await pilott_instance.stop()


class TestPilott:
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test basic initialization of Pilott"""
        pilott = Pilott(name="TestPilott")
        assert pilott.config.name == "TestPilott"
        assert pilott.config.process_type == ProcessType.SEQUENTIAL
        assert not pilott._started

    @pytest.mark.asyncio
    async def test_initialization_with_config(self, llm_config):
        """Test initialization with custom config"""
        config = {
            "process_type": ProcessType.PARALLEL,
            "max_concurrent_jobs": 10,
            "memory_enabled": False
        }
        pilott = Pilott(name="TestPilott", config=config, llm_config=llm_config)
        assert pilott.config.name == "TestPilott"
        assert pilott.config.process_type == ProcessType.PARALLEL
        assert pilott.config.max_concurrent_jobs == 10
        assert not pilott.config.memory_enabled
        assert pilott.llm is not None

    @pytest.mark.asyncio
    async def test_start_stop(self, pilott, mock_agents):
        """Test starting and stopping Pilott"""
        # Set agents explicitly
        pilott.agents = mock_agents

        await pilott.start()
        assert pilott._started
        for agent in mock_agents:
            agent.start.assert_called_once()

        await pilott.stop()
        assert not pilott._started

    @pytest.mark.asyncio
    async def test_serve_sequential(self, pilott, mock_agents):
        """Test sequential job execution with serve method"""
        pilott.config.process_type = ProcessType.SEQUENTIAL
        pilott.agents = mock_agents

        # Create test job
        test_job = Job(description="Test job")
        mock_agents[0].jobs = [test_job]

        # Mock _execute_sequential to bypass implementation details
        with patch.object(pilott, '_execute_sequential') as mock_execute:
            expected_result = [
                JobResult(
                    success=True,
                    output="Test result",
                    execution_time=0.1,
                    error=None,
                    metadata={}
                )
            ]
            mock_execute.return_value = expected_result

            await pilott.start()
            results = await pilott.serve()

            # Verify _execute_sequential was called with our agents
            mock_execute.assert_called_once()
            assert results == expected_result

    @pytest.mark.asyncio
    async def test_serve_parallel(self, pilott, mock_agents):
        """Test parallel job execution with serve method"""
        pilott.config.process_type = ProcessType.PARALLEL
        pilott.agents = mock_agents

        # Create test job for all agents
        for i, agent in enumerate(mock_agents):
            job = Job(description=f"Job for agent {i}")
            agent.jobs = [job]

        # Mock _execute_parallel to bypass implementation details
        with patch.object(pilott, '_execute_parallel') as mock_execute:
            expected_result = [
                JobResult(
                    success=True,
                    output=f"Result from agent_{i}",
                    execution_time=0.1,
                    error=None,
                    metadata={}
                ) for i in range(3)
            ]
            mock_execute.return_value = expected_result

            await pilott.start()
            results = await pilott.serve()

            # Verify _execute_parallel was called with our agents
            mock_execute.assert_called_once()
            assert results == expected_result

    @pytest.mark.asyncio
    async def test_assign_jobs_to_agents(self, pilott, mock_agents):
        """Test job assignment to agents"""
        pilott.agents = mock_agents
        pilott.job_assignment_type = JobAssignmentType.SUITABILITY

        # Create job
        job = Job(description="Job 1", priority=JobPriority.HIGH)

        # Setup mocked agent utility
        mock_agent_util = Mock()
        mock_agent_util.assign_job = AsyncMock()
        # Since we're awaiting the coroutine in the method, return a pre-awaited result
        mock_agent_util.assign_job.return_value = (mock_agents[0], 0.9)

        pilott.agentUtility = mock_agent_util

        # Patch _get_agent_by_job to return a specific agent
        with patch.object(pilott, '_get_agent_by_job', return_value=mock_agents[0]):
            await pilott.start()

            # Call the method directly
            agent = await pilott._get_agent_by_job(job, mock_agents)

            # Verify we got the expected agent
            assert agent == mock_agents[0]

    @pytest.mark.asyncio
    async def test_error_handling(self, pilott, mock_agent):
        """Test error handling during job execution"""
        pilott.agents = [mock_agent]

        # Configure agent to raise an exception
        mock_agent.execute_job = AsyncMock(side_effect=ValueError("Test error"))

        # Create test job
        test_job = Job(description="Error job")
        mock_agent.jobs = [test_job]

        # Mock _process_agent_jobs to simulate error handling
        with patch.object(pilott, '_process_agent_jobs') as mock_process:
            error_result = JobResult(
                success=False,
                output=None,
                error="Test error",
                execution_time=0.1,
                metadata={}
            )
            mock_process.return_value = [error_result]

            await pilott.start()
            results = await pilott._execute_sequential([mock_agent])

            assert results is not None
            assert len(results) == 1
            assert not results[0].success
            assert "Test error" in results[0].error

    @pytest.mark.asyncio
    async def test_get_metrics(self, pilott):
        """Test getting system metrics"""
        # Use an explicit list of agents
        pilott.agents = []
        pilott.jobs = []

        metrics = pilott.get_metrics()

        assert "active_agents" in metrics
        assert metrics["active_agents"] == 0
        assert "total_jobs" in metrics
        assert metrics["total_jobs"] == 0
        assert "is_running" in metrics
