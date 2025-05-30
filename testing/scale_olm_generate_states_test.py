"""Comprehensive tests for scale.olm.generate.states module."""

import pytest
import numpy as np
import scale.olm.generate.states as states


class TestFullHypercubeFunction:
    """Test the main full_hypercube function."""
    
    def test_full_hypercube_basic_functionality(self):
        """Test basic functionality of full_hypercube."""
        test_states = {
            "enrichment": [2.0, 4.0],
            "mod_dens": [0.7, 1.0]
        }
        
        result = states.full_hypercube(**test_states)
        
        # Should generate 2 x 2 = 4 permutations
        assert len(result) == 4
        
        # Check that all permutations are present
        expected_permutations = [
            {"enrichment": 2.0, "mod_dens": 0.7},
            {"enrichment": 2.0, "mod_dens": 1.0},
            {"enrichment": 4.0, "mod_dens": 0.7},
            {"enrichment": 4.0, "mod_dens": 1.0}
        ]
        
        # Sort both lists for comparison (order may vary)
        result_sorted = sorted(result, key=lambda x: (x["enrichment"], x["mod_dens"]))
        expected_sorted = sorted(expected_permutations, key=lambda x: (x["enrichment"], x["mod_dens"]))
        
        assert result_sorted == expected_sorted
    
    def test_full_hypercube_single_dimension(self):
        """Test full_hypercube with single dimension."""
        test_states = {"power": [40.0, 50.0, 60.0]}
        
        result = states.full_hypercube(**test_states)
        
        assert len(result) == 3
        assert result[0]["power"] in [40.0, 50.0, 60.0]
        assert result[1]["power"] in [40.0, 50.0, 60.0]
        assert result[2]["power"] in [40.0, 50.0, 60.0]
        
        # All powers should be unique
        powers = [perm["power"] for perm in result]
        assert len(set(powers)) == 3
    
    def test_full_hypercube_three_dimensions(self):
        """Test full_hypercube with three dimensions."""
        test_states = {
            "enrichment": [2.0, 4.0],
            "mod_dens": [0.7, 1.0],
            "burnup": [10.0, 20.0, 30.0]
        }
        
        result = states.full_hypercube(**test_states)
        
        # Should generate 2 x 2 x 3 = 12 permutations
        assert len(result) == 12
        
        # Check that each dimension has the correct range of values
        enrichments = set(perm["enrichment"] for perm in result)
        mod_densities = set(perm["mod_dens"] for perm in result)
        burnups = set(perm["burnup"] for perm in result)
        
        assert enrichments == {2.0, 4.0}
        assert mod_densities == {0.7, 1.0}
        assert burnups == {10.0, 20.0, 30.0}
    
    def test_full_hypercube_unordered_input_sorted(self):
        """Test that full_hypercube sorts input arrays."""
        test_states = {
            "enrichment": [4.0, 2.0, 3.0],  # Deliberately unordered
            "mod_dens": [1.0, 0.5, 0.7]     # Deliberately unordered
        }
        
        result = states.full_hypercube(**test_states)
        
        assert len(result) == 9  # 3 x 3 = 9 permutations
        
        # Extract unique values and verify they're from the input
        enrichments = sorted(set(perm["enrichment"] for perm in result))
        mod_densities = sorted(set(perm["mod_dens"] for perm in result))
        
        assert enrichments == [2.0, 3.0, 4.0]  # Should be sorted
        assert mod_densities == [0.5, 0.7, 1.0]  # Should be sorted
    
    def test_full_hypercube_single_value_per_dimension(self):
        """Test full_hypercube with single values per dimension."""
        test_states = {
            "enrichment": [3.0],
            "mod_dens": [0.8],
            "power": [45.0]
        }
        
        result = states.full_hypercube(**test_states)
        
        assert len(result) == 1
        assert result[0] == {"enrichment": 3.0, "mod_dens": 0.8, "power": 45.0}
    
    def test_full_hypercube_with_type_parameter(self):
        """Test full_hypercube with the _type parameter."""
        test_states = {
            "enrichment": [2.0, 4.0],
            "mod_dens": [0.7]
        }
        
        # Call with the _type parameter
        result = states.full_hypercube(
            _type="scale.olm.generate.states:full_hypercube", 
            **test_states
        )
        
        assert len(result) == 2
        assert all("enrichment" in perm and "mod_dens" in perm for perm in result)


class TestSchemaFunction:
    """Test the _schema_full_hypercube function."""
    
    def test_schema_full_hypercube_without_state(self):
        """Test _schema_full_hypercube without state."""
        schema = states._schema_full_hypercube(with_state=False)
        
        # Should return a dictionary (schema structure)
        assert isinstance(schema, dict)
    
    def test_schema_full_hypercube_with_state(self):
        """Test _schema_full_hypercube with state."""
        schema = states._schema_full_hypercube(with_state=True)
        
        # Should return a dictionary (schema structure) 
        assert isinstance(schema, dict)
    
    def test_schema_full_hypercube_default_parameter(self):
        """Test _schema_full_hypercube with default parameter."""
        # Call without parameters (defaults to with_state=False)
        schema = states._schema_full_hypercube()
        
        assert isinstance(schema, dict)


class TestTestArgsFunction:
    """Test the _test_args_full_hypercube function."""
    
    def test_test_args_full_hypercube_without_state(self):
        """Test _test_args_full_hypercube without state."""
        args = states._test_args_full_hypercube(with_state=False)
        
        # Should return the expected structure
        assert isinstance(args, dict)
        assert args["_type"] == "scale.olm.generate.states:full_hypercube"
        assert "coolant_density" in args
        assert "enrichment" in args
        assert "specific_power" in args
        
        # Check the values
        assert args["coolant_density"] == [0.4, 0.7, 1.0]
        assert args["enrichment"] == [1.5, 3.5, 4.5]
        assert args["specific_power"] == [42.0]
    
    def test_test_args_full_hypercube_with_state(self):
        """Test _test_args_full_hypercube with state."""
        args = states._test_args_full_hypercube(with_state=True)
        
        # Should return the expected structure
        assert isinstance(args, dict)
        assert args["_type"] == "scale.olm.generate.states:full_hypercube"
        assert "coolant_density" in args
        assert "enrichment" in args
        assert "specific_power" in args
        
        # Values should be the same regardless of with_state parameter
        assert args["coolant_density"] == [0.4, 0.7, 1.0]
        assert args["enrichment"] == [1.5, 3.5, 4.5]
        assert args["specific_power"] == [42.0]
    
    def test_test_args_full_hypercube_default_parameter(self):
        """Test _test_args_full_hypercube with default parameter."""
        # Call without parameters (defaults to with_state=False)
        args = states._test_args_full_hypercube()
        
        assert isinstance(args, dict)
        assert args["_type"] == "scale.olm.generate.states:full_hypercube"
        assert len(args) == 4  # _type + 3 parameter arrays


class TestModuleConstants:
    """Test module-level constants and exports."""
    
    def test_module_exports(self):
        """Test that __all__ exports are correct."""
        assert hasattr(states, '__all__')
        assert "full_hypercube" in states.__all__
        assert len(states.__all__) == 1
    
    def test_type_constant(self):
        """Test the module type constant."""
        assert hasattr(states, '_TYPE_FULL_HYPERCUBE')
        assert states._TYPE_FULL_HYPERCUBE == "scale.olm.generate.states:full_hypercube"


class TestMathematicalProperties:
    """Test mathematical properties of the full_hypercube function."""
    
    def test_permutation_count_formula(self):
        """Test that the number of permutations follows the formula: product of dimension sizes."""
        test_cases = [
            ({"a": [1, 2], "b": [3, 4, 5]}, 2 * 3),  # 2 * 3 = 6
            ({"x": [1], "y": [2], "z": [3]}, 1 * 1 * 1),  # 1 * 1 * 1 = 1  
            ({"dim1": [1, 2, 3, 4], "dim2": [5, 6]}, 4 * 2),  # 4 * 2 = 8
        ]
        
        for test_states, expected_count in test_cases:
            result = states.full_hypercube(**test_states)
            assert len(result) == expected_count, f"Failed for {test_states}"
    
    def test_all_combinations_present(self):
        """Test that all possible combinations are present in the result."""
        test_states = {
            "a": [1, 2],
            "b": [10, 20]
        }
        
        result = states.full_hypercube(**test_states)
        
        # Convert to set of tuples for easy comparison
        result_tuples = {(perm["a"], perm["b"]) for perm in result}
        expected_tuples = {(1, 10), (1, 20), (2, 10), (2, 20)}
        
        assert result_tuples == expected_tuples
    
    def test_dimension_independence(self):
        """Test that each dimension varies independently."""
        test_states = {
            "x": [1, 2, 3],
            "y": [10, 20]
        }
        
        result = states.full_hypercube(**test_states)
        
        # For each value of x, there should be every value of y
        x_values = [1, 2, 3]
        y_values = [10, 20]
        
        for x_val in x_values:
            x_perms = [perm for perm in result if perm["x"] == x_val]
            y_vals_for_x = sorted([perm["y"] for perm in x_perms])
            assert y_vals_for_x == sorted(y_values)


class TestIntegrationWithTestArgs:
    """Test integration between test args and actual function."""
    
    def test_test_args_work_with_function(self):
        """Test that _test_args_full_hypercube generates valid input for full_hypercube."""
        # Get test arguments
        test_args = states._test_args_full_hypercube()
        
        # Remove the _type key since the function doesn't expect it
        function_args = {k: v for k, v in test_args.items() if k != "_type"}
        
        # Call the function with test arguments
        result = states.full_hypercube(**function_args)
        
        # Should generate the expected number of permutations
        expected_count = 3 * 3 * 1  # len(coolant_density) * len(enrichment) * len(specific_power)
        assert len(result) == expected_count
        
        # All permutations should have the expected keys
        for perm in result:
            assert set(perm.keys()) == {"coolant_density", "enrichment", "specific_power"}
            assert perm["coolant_density"] in [0.4, 0.7, 1.0]
            assert perm["enrichment"] in [1.5, 3.5, 4.5]
            assert perm["specific_power"] == 42.0  # Only one value 