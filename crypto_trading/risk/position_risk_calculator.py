def calculate_position_size(asset_volatility, portfolio_value, risk_tolerance, current_exposure):
    """
    Calculate the maximum position size based on asset volatility, portfolio risk tolerance, and current exposure.

    Parameters:
    - asset_volatility (float): The volatility of the asset, typically expressed as a percentage (e.g., 0.05 for 5%).
    - portfolio_value (float): The total value of the portfolio.
    - risk_tolerance (float): The maximum percentage of the portfolio value that the investor is willing to risk (e.g., 0.02 for 2%).
    - current_exposure (float): The current exposure to the asset in the portfolio, expressed as a percentage of the portfolio value.

    Returns:
    - max_position_size_absolute (float): The maximum position size in absolute terms (currency units).
    - max_position_size_percentage (float): The maximum position size as a percentage of the portfolio value.
    """
    # Calculate the maximum risk amount in currency terms
    max_risk_amount = portfolio_value * risk_tolerance

    # Calculate the maximum position size in absolute terms
    max_position_size_absolute = max_risk_amount / asset_volatility

    # Calculate the maximum position size as a percentage of the portfolio value
    max_position_size_percentage = (max_position_size_absolute / portfolio_value) * 100

    # Adjust for current exposure
    max_position_size_absolute -= current_exposure * portfolio_value
    max_position_size_percentage -= current_exposure * 100

    return max_position_size_absolute, max_position_size_percentage


# Test cases
def test_calculate_position_size():
    # Test case 1
    asset_volatility = 0.05
    portfolio_value = 100000
    risk_tolerance = 0.02
    current_exposure = 0.01
    result = calculate_position_size(asset_volatility, portfolio_value, risk_tolerance, current_exposure)
    assert result == (18000.0, 18.0), f"Test case 1 failed: {result}"

    # Test case 2
    asset_volatility = 0.1
    portfolio_value = 200000
    risk_tolerance = 0.01
    current_exposure = 0.02
    result = calculate_position_size(asset_volatility, portfolio_value, risk_tolerance, current_exposure)
    assert result == (16000.0, 8.0), f"Test case 2 failed: {result}"

    # Test case 3
    asset_volatility = 0.02
    portfolio_value = 50000
    risk_tolerance = 0.05
    current_exposure = 0.0
    result = calculate_position_size(asset_volatility, portfolio_value, risk_tolerance, current_exposure)
    assert result == (125000.0, 250.0), f"Test case 3 failed: {result}"

    print("All test cases passed!")


if __name__ == "__main__":
    test_calculate_position_size()