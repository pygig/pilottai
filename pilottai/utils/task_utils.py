from typing import Union, List, Any, Dict, Optional
from datetime import datetime

from pilottai.core.task import Task, TaskResult


class TaskUtility:
    """
    Utility class for working with tasks, providing helper methods
    for task conversion, validation, and common operations.
    """

    @staticmethod
    def to_task(task_input: Union[str, Dict, Task]) -> List[Task]:
        """
        Convert a string, dictionary, or Task object to a Task instance.

        Args:
            task_input: A string (treated as description), dictionary, or Task object

        Returns:
            Task: A properly instantiated Task object

        Raises:
            ValueError: If the input cannot be converted to a Task
        """
        if isinstance(task_input, Task):
            return task_input

        elif isinstance(task_input, str):
            return Task(description=task_input)

        elif isinstance(task_input, dict):
            # Ensure the dictionary has at least a description
            if "description" not in task_input:
                raise ValueError("Task dictionary must contain a 'description' field")

            return Task(**task_input)

        else:
            raise ValueError(
                f"Cannot convert {type(task_input)} to Task. Must be a string, dictionary, or Task object.")

    @staticmethod
    def to_task_list(task_inputs: Union[str, Dict, Task, List[str], List[Dict], List[Task]]) -> List[Task]:
        """
        Convert various input formats to a list of Task objects.

        Args:
            task_inputs: A single task or list of tasks in various formats

        Returns:
            List[Task]: A list of properly instantiated Task objects
        """
        # Handle single items
        if isinstance(task_inputs, (str, dict, Task)):
            return TaskUtility.to_task(task_inputs)

        # Handle lists
        elif isinstance(task_inputs, list):
            return [TaskUtility.to_task(item) for item in task_inputs]

        else:
            raise ValueError(f"Cannot convert {type(task_inputs)} to a list of Tasks")

    @staticmethod
    def is_task_object(task_input: Any) -> bool:
        """
        Check if the input is a Task object.

        Args:
            task_input: Any input to check

        Returns:
            bool: True if the input is a Task object, False otherwise
        """
        return isinstance(task_input, Task)

    @staticmethod
    def get_task_type(task_input: Any) -> str:
        """
        Get the type of the task input.

        Args:
            task_input: Any input to check

        Returns:
            str: The type of the task input ('task', 'str', 'dict', or 'unknown')
        """
        if isinstance(task_input, Task):
            return 'task'
        elif isinstance(task_input, str):
            return 'str'
        elif isinstance(task_input, dict):
            return 'dict'
        else:
            return 'unknown'

    @staticmethod
    def create_empty_result(task: Task, error: Optional[str] = None) -> TaskResult:
        """
        Create an empty (failed) result for a task.

        Args:
            task: The task for which to create a result
            error: Optional error message

        Returns:
            TaskResult: A failed task result
        """
        return TaskResult(
            success=False,
            output=None,
            error=error or "Task execution failed",
            execution_time=0.0,
            metadata={
                "task_id": task.id,
                "created_at": datetime.now().isoformat()
            }
        )

    @staticmethod
    def merge_task_results(results: List[TaskResult]) -> TaskResult:
        """
        Merge multiple task results into a single result.

        Args:
            results: List of task results to merge

        Returns:
            TaskResult: A consolidated task result
        """
        if not results:
            return TaskResult(
                success=True,
                output="No tasks executed",
                error=None,
                execution_time=0.0
            )

        # Calculate overall success and total execution time
        overall_success = all(result.success for result in results)
        total_execution_time = sum(result.execution_time for result in results)

        # Combine outputs and errors
        outputs = []
        errors = []

        for i, result in enumerate(results):
            if result.output:
                outputs.append(f"Task {i + 1} output: {result.output}")
            if result.error:
                errors.append(f"Task {i + 1} error: {result.error}")

        combined_output = "\n".join(outputs) if outputs else None
        combined_error = "\n".join(errors) if errors else None

        return TaskResult(
            success=overall_success,
            output=combined_output,
            error=combined_error,
            execution_time=total_execution_time,
            metadata={
                "result_count": len(results),
                "success_count": sum(1 for r in results if r.success),
                "fail_count": sum(1 for r in results if not r.success)
            }
        )
